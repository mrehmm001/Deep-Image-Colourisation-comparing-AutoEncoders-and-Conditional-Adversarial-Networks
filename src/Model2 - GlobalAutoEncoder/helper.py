import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

IMG_WIDTH = 256
IMG_HEIGHT = 256

#HELPER FUNCTIONS=======================================================================================================================

class FusionLayer(Layer):
    def call(self, inputs, mask=None):
        imgs, embs = inputs
        # reshaped_shape = imgs.shape[:3].concatenate(embs.shape[1])
        reshaped_shape = (tf.shape(imgs)[0],imgs.shape[1],imgs.shape[1],embs.shape[1])
        embs = K.repeat(embs, imgs.shape[1] * imgs.shape[2])
        embs = K.reshape(embs, tf.stack(reshaped_shape))
        return K.concatenate([imgs, embs], axis=3)

    def compute_output_shape(self, input_shapes):
        # Must have 2 tensors as input
        assert input_shapes and len(input_shapes) == 2
        imgs_shape, embs_shape = input_shapes

        # The batch size of the two tensors must match
        assert imgs_shape[0] == embs_shape[0]

        # (batch_size, width, height, embedding_len + depth)
        return imgs_shape[:3] + (imgs_shape[3] + embs_shape[1],)




def getGlobal_encoder(model_input):
  from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
  img_conc = layers.Concatenate()([model_input, model_input, model_input])  
  inputModel = InceptionResNetV2(weights='imagenet',
                      include_top=True,
                      input_tensor=img_conc)
  inputModel.trainable = False
  return inputModel.output


def rgb_to_xyz(input, name=None):
    """
    Convert a RGB image to CIE XYZ.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    kernel = tf.constant(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        input.dtype,
    )
    value = tf.where(
        tf.math.greater(input, 0.04045),
        tf.math.pow((input + 0.055) / 1.055, 2.4),
        input / 12.92,
    )
    return tf.tensordot(value, tf.transpose(kernel), axes=((-1,), (0,)))


def rgb_to_lab(input, illuminant="D65", observer="2", name=None):
    """
    Convert a RGB image to CIE LAB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
      observer : {"2", "10"}, optional
        The aperture angle of the observer.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = tf.constant(illuminants[illuminant.upper()][observer], input.dtype)

    xyz = rgb_to_xyz(input)

    xyz = xyz / coords

    xyz = tf.where(
        tf.math.greater(xyz, 0.008856),
        tf.math.pow(xyz, 1.0 / 3.0),
        xyz * 7.787 + 16.0 / 116.0,
    )

    xyz = tf.unstack(xyz, axis=-1)
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Vector scaling
    l = (y * 116.0) - 16.0
    a = (x - y) * 500.0
    b = (y - z) * 200.0

    return tf.stack([l, a, b], axis=-1)



def load(image):
  # Read and decode an image file to a uint8 tensor
  image=image/256
  image = rgb_to_lab(image)
  image = tf.image.resize(image,(IMG_WIDTH,IMG_HEIGHT))
  batch = tf.shape(image)[0]
  # coloured = tf.image.resize(image[:,:,:,1:],(IMG_WIDTH,IMG_HEIGHT))
  coloured = image[:,:,:,1:]
  grayscale = image[:,:,:,0]
  grayscale=tf.reshape(grayscale,(batch,IMG_WIDTH,IMG_HEIGHT,1))
  # grayscale = tf.image.resize(grayscale,(IMG_WIDTH,IMG_HEIGHT))

  # Convert both images to float32 tensors
  grayscale = tf.cast(grayscale, tf.float32)
  coloured = tf.cast(coloured, tf.float32)
  return grayscale, coloured  

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image  

# Normalizing the images to [0, 1]
def normalize(input_image, real_image):
  # input_image = (input_image / 127.5) - 1
  # real_image = (real_image / 127.5) - 1
  input_image = input_image/100
  real_image = real_image/127

  return input_image, real_image

@tf.function
def random_crop(input_image, real_image):
  input_image = tf.concat([input_image, input_image], axis=-1)
  batch = tf.shape(input_image)[0]
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2,batch, IMG_HEIGHT, IMG_WIDTH,2])
  input = cropped_image[0][:,:,:,0]
  input = tf.reshape(input,[tf.shape(input)[0],tf.shape(input)[1],tf.shape(input)[2],1])
  target = cropped_image[1]
  print(input.shape)
  print(target.shape)

  return input, target

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)
  import random
  if random.uniform(0, 1)> 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
   


# Normalizing the images to [0, 1]
def normalize(input_image, real_image):
  # input_image = (input_image / 127.5) - 1
  # real_image = (real_image / 127.5) - 1
  input_image = input_image/100
  real_image = real_image/127

  return input_image, real_image


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  # input_image, real_image = random_jitter(input_image, real_image)
  # input_image, real_image = resize(input_image, real_image,
  #                                  IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image  


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def preprocessor(generator,function):
  for batch in generator:
    images,_ = batch
    yield function(images)

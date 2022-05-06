import tensorflow as tf
import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

IMG_WIDTH = 256
IMG_HEIGHT = 256

#HELPER FUNCTIONS=======================================================================================================================

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.d_accuracy = tf.keras.metrics.Accuracy(name="d_acc")
        self.g_accuracy = tf.keras.metrics.Accuracy(name="g_acc")
    
    def call(self, inputs):
       print("") 


    def compile(self, d_optimizer, g_optimizer, g_loss,d_loss):
        super(GAN, self).compile()
        self.discriminator_optimizer = d_optimizer
        self.generator_optimizer = g_optimizer
        self.generator_loss = g_loss
        self.discriminator_loss = d_loss

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric,self.d_accuracy, self.g_accuracy]

    def getAccuracy(pred,tar):
      g = tf.Graph()
      with g.as_default():
        logits = example_input.numpy()
        labels = example_target.numpy()
        acc, acc_op = tf.compat.v1.metrics.accuracy(logits, labels)
        global_init = tf.compat.v1.global_variables_initializer()
        local_init = tf.compat.v1.local_variables_initializer()
      sess = tf.compat.v1.Session(graph=g)
      sess.run([global_init, local_init])
      accuracy = (sess.run([acc]))[0]
      return accuracy    

    def train_step(self,dataset):
      input_image, target = dataset
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = self.generator(input_image, training=True)

        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)
        # accuracy= getAccuracy(gen_output,target)
        # self.g_accuracy.update_state(tf.round(input_image),tf.round(target))

        gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                  self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                              self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  self.discriminator.trainable_variables))
      self.d_loss_metric.update_state(disc_loss)
      self.g_loss_metric.update_state(gen_total_loss)    
      # accuracy = K.mean(K.equal(target, K.round(gen_output)))*100
      self.g_accuracy.update_state(target,K.round(gen_output))


      return {"d_loss": self.d_loss_metric.result(), 
              "g_loss": self.g_loss_metric.result()}






def load(image):
  # Read and decode an image file to a uint8 tensor
  coloured = tf.image.resize(image,(IMG_WIDTH,IMG_HEIGHT))
  grayscale = tf.image.resize(tf.image.rgb_to_grayscale(image),(IMG_WIDTH,IMG_HEIGHT))
  grayscale = tf.concat([grayscale,grayscale,grayscale],-1)
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
  input_image = input_image/256
  real_image = real_image/256

  return input_image, real_image

def random_crop(input_image, real_image):
  batch = tf.shape(input_image)[0]
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2,batch, IMG_HEIGHT, IMG_WIDTH, 3])
  print(cropped_image)

  return cropped_image[0], cropped_image[1]

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)
  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
   

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result  


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image  


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def getAccuracy(pred,tar):
  accuracy = K.mean(K.equal(tar, pred))*100
  return accuracy

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  accuracy = getAccuracy(prediction,tar)
  print("accuracy: ",np.array(accuracy))
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()  


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
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

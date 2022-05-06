#IMPORTS=======================================================================================================================
import tensorflow as tf
import os
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Activation,Reshape, UpSampling2D,Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from skimage.io import imshow
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
# import tensorflow_io as tfio
from helper import *

# The facade training set consist of 400 images
BUFFER_SIZE = 400

BATCH_SIZE = 64

IMAGE_HEIGHT_WIDTH=256


#Check GPU version and tensorflow
print("Is GPU being used? ",len(tf.config.list_physical_devices('GPU'))>0)
print("TensorFlow version: ",tf.__version__)


#Specify the train & val link and then you're set to go!
TRAIN_PATH = "" 
VAL_PATH = "" 

#PIPELINE=======================================================================================================================

train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    batch_size=BATCH_SIZE)

train_dataset = preprocessor(train_generator,load_image_train)

val_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)
val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    batch_size=BATCH_SIZE)

val_dataset = preprocessor(val_generator,load_image_test)


#MODEL==================================================================================================
#Construct the model
model = Sequential()

#Encoder
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(IMAGE_HEIGHT_WIDTH, IMAGE_HEIGHT_WIDTH, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

#Decoder
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))


#Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse' , metrics=['accuracy'])





#TRAIN===========================================================================================
train_steps = train_generator.samples // BATCH_SIZE
val_steps = val_generator.samples//BATCH_SIZE
checkpoint_filepath = './checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history=model.fit(train_dataset, epochs = 50,steps_per_epoch=train_steps,shuffle=True,callbacks=[model_checkpoint_callback],validation_steps = val_steps, validation_data=val_dataset)
print("Model finished training")

#SAVE==========================================================================================
model.save("model.h5")
np.save("history.npy",history.history)
print("Model has been saved")

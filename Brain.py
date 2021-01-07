import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random
import os
from PIL import Image
import h5py

print(os.listdir("/home/solibot/Documents/CropCure/TEA Dataset"))


blight_dir = os.path.join('/home/solibot/Documents/CropCure/TEA Dataset/data/test/Tea leaf blight')
spot_dir = os.path.join('/home/solibot/Documents/CropCure/TEA Dataset/data/test/Tea red leaf spot')
scab_dir = os.path.join('/home/solibot/Documents/CropCure/TEA Dataset/data/test/Tea red scab')

print('Total training Tea Blight images:', len(os.listdir(blight_dir)))
print('Total training Tea Red Spot images:', len(os.listdir(spot_dir)))
print('Total training Tea Red Scab images:', len(os.listdir(scab_dir)))

blight_files = os.listdir(blight_dir)
print(blight_files[:10])

spot_files = os.listdir(spot_dir)
print(spot_files[:10])

scab_files = os.listdir(scab_dir)
print(scab_files[:10])


pic_index = 2

next_blight = [os.path.join(blight_dir, fname) 
                for fname in blight_files[pic_index-2:pic_index]]
next_spot = [os.path.join(spot_dir, fname) 
                for fname in spot_files[pic_index-2:pic_index]]
next_scab = [os.path.join(scab_dir, fname) 
                for fname in scab_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_blight+next_spot+next_scab):
    img = mpimg.imread(img_path)


TRAINING_DIR = "/home/solibot/Documents/CropCure/TEA Dataset/data/train"
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "/home/solibot/Documents/CropCure/TEA Dataset/data/test"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical')


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(train_generator, epochs=10, validation_data = validation_generator, verbose = 1)

model.save('neural_network_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

print(epochs)

print("Accuracy: \n", acc,"\n\n")
print("Validation Accuracy: \n", val_acc, "\n\n")
print("Neural Network Loss: \n", loss, "\n\n")
print("Neural Network Validation Loss: \n", loss, "\n\n")

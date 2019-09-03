# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:17:59 2019

@author: Navnit Singh
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:\\ML\\Convolutional Neural Network\\dataset\\training_set',
                                                 target_size = (64, 64), # image dimension
                                                 batch_size = 32, # number of images sent to the CPU (or GPU)
                                                 class_mode = 'binary') # binary: cat/dog
 
# test set
test_set = test_datagen.flow_from_directory('D:\\ML\\Convolutional Neural Network\\dataset\\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
 
classifier.fit_generator( training_set,
                         steps_per_epoch = training_set.samples, # 8000
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = test_set.samples) # 2000


import numpy as np
from keras.preprocessing import image
test_image1 = image.load_img('C:\\Users\\Navnit Singh\\Desktop\\cat.jpg', target_size = (64, 64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)
result1 = classifier.predict(test_image1)
training_set.class_indices
if result1[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'



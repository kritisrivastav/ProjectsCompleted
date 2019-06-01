# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:58:52 2019

@author: HP
"""
import numpy as np
import pickle
import cv2
import os
from keras import optimizers#for compiling a keras model
from keras.models import Sequential#linear stack of layers
from keras.layers import Dense#Dense is a name for fully connected/linear layer in keras
from keras.layers import Dropout#Dropout is a technique where randomly selected neurons are ignored during training.
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint#Adding checkpoints to your trainning process
from keras import backend as K#Keras does not do simple level operations like convolution so tensorflow or theano does it
K.set_image_dim_ordering('tf')#"tf" format means that the convolutional kernels will have the shape (rows, cols, input_depth, depth)

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(os.listdir('gestures/'))

image_x, image_y = get_image_size()

def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()#creates a linear stack of layers
	model.add(Conv2D(32, (5,5), input_shape=(image_x, image_y, 1), activation='relu'))# 5*5 is filter size with random weights dealing with a single graysacle image
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))#it depends on the size of data used for training.
	model.add(Dropout(0.4))#Fraction of the input units to drop.
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-4)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras2.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint1]# the model checkpoints will be saved with the epoch number and the validation loss in the filename.
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)
	test_labels = np_utils.to_categorical(test_labels)

	model, callbacks_list = cnn_model()
	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50, batch_size=100, callbacks=callbacks_list)
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')

train()

#Downloading video from Youtube

from pytube import YouTube as yt
video_link="https://www.youtube.com/watch?v=YWcu_8xPSs8"
vid=yt(video_link)
stream = vid.streams.first()  #Getting the first stream
stream.download()

'''Now we have our data in the form of a video which is nothing 
but a group of frames( images). Since we are going to solve this problem using image classification.
We need to extract the images from the video'''

import cv2
'''Opens the video File'''
cap=cv2.VideoCapture('D:\\Interviews\\video\\1.mp4')
i=0
image='x'
ret=1
while ret:
    ret,frame=cap.read()
    if ret==False:
        break
    cv2.imwrite("frame%d.jpg" % i,frame)
    i+=1
cap.release()
cv2.destroyAllWindows()

#After this the video will be divided into individual frames.In this problem I have taken only two class,amitabh or no amitabh
'''Input data and preprocessing'''
'''We have data in the form of images.To prepare this data for the neural network,We need to do some preprocessing'''
from tqdm import tqdm
import cv2
import os
import numpy as np
img_path ='D:\image'
class1_data = []
class2_data = []
for classes in os.listdir(img_path):
        fin_path = os.path.join(img_path, classes)
        for fin_classes in tqdm(os.listdir(fin_path)):
            img = cv2.imread(os.path.join(fin_path, fin_classes))
            img = cv2.resize(img, (224,224))
            img = img/255.
            if classes == 'amitabh':
                class1_data.append(img)
            else:
                class2_data.append(img)
 
class1_data = np.array(class1_data)
class2_data = np.array(class2_data)

'''Since the number of images we are using here is very less,Using transfer learning we can use features generated by a model
trained on a large dataset into our model.Here we will use VGG16 model trained on “imagenet” dataset.For this,
 we are using tensorflow high-level API Keras'''

import keras
from keras.applications import VGG16
vgg_model = VGG16(include_top=False, weights='imagenet')

'''Now we will pass our input data to vgg_model and generate the features.'''
vgg_class1 = vgg_model.predict(class1_data)
vgg_class2 = vgg_model.predict(class2_data)

'''Since we are not including fully connected layers from VGG16 model, we need to create a model 
with some fully connected layers and an output layer with 1 class, either “amitabh” or “No amitabh”. Output 
features from VGG16 model will be having shape 7*7*512,
 which will be input shape for our model. Here I am also using dropout layer to make model less over-fit.'''
 
 
from keras.layers import Input, Dense, Dropout
from keras.models import Model
inputs = Input(shape=(7*7*512,))
dense1 = Dense(1024, activation = 'relu')(inputs)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(dense2)
outputs = Dense(1, activation = 'sigmoid')(drop2)
 
model = Model(inputs, outputs)
model.summary()

'''Splitting Data into Train and Validation'''
train_data = np.concatenate((vgg_class1[:50], vgg_class2[:50]), axis = 0)
train_data = train_data.reshape(train_data.shape[0],7*7*512)
 
valid_data = np.concatenate((vgg_class1[50:], vgg_class2[50:]), axis = 0)
valid_data = valid_data.reshape(valid_data.shape[0],7*7*512)

	
train_label = np.array([0]*vgg_class1[:50].shape[0] + [1]*vgg_class2[:50].shape[0])
valid_label = np.array([0]*vgg_class1[50:].shape[0] + [1]*vgg_class2[50:].shape[0])

'''Now we will be training the network..Here, we will use stochastic gradient descent as an optimizer and 
binary cross-entropy as our loss function.
 We are also going to save our checkpoint for the best model according to it’s validation dataset accuracy.'''

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train_data, train_label, epochs = 10, batch_size = 64, validation_data = (valid_data, valid_label), verbose = 2, callbacks = callbacks_list)

'''Calculating Screen Time'''


import os
import numpy as np

amitabh_images = []
no_amitabh_images = []

test_path = 'D:\image'

for test in tqdm(os.listdir(test_path)):
    test_img = cv2.imread(os.path.join(test_path, test))
    test_img = cv2.resize(test_img, (224,224))
    test_img = test_img/255.
    test_img = np.expand_dims(test_img, 0)
    pred_img = vgg_model.predict(test_img)
    pred_feat = pred_img.reshape(1, 7*7*512)
    out_class = model.predict(pred_feat)
    if out_class < 0.5:
        amitabh_images.append(out_class)
    else:
        no_amitabh_images.append(out_class)




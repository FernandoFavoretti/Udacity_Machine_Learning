
# coding: utf-8

# In[62]:

import numpy as np
import pandas as pd
from PIL import Image
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from glob import glob


# In[63]:

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy


# In[25]:

images_dir = "train/"
num_files = len(glob(os.path.join(images_dir, '*.jpg')))
print(num_files)


# In[41]:

img_size = 64
all_X = np.zeros((num_files, img_size, img_size, 3), dtype='float64')
all_y = np.zeros(num_files)


# In[26]:

def transform_images(img, width, height):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    return img


# In[42]:

def make_train_set(data_folder):
    i = 0
    image_filenames = os.listdir(data_folder)
    for image_filename in image_filenames:
        image_path = os.path.join(data_folder, image_filename)
        img = cv2.imread(image_path)
        img = transform_images(img, img_size, img_size)
        all_X[i] = np.array(img)
        all_y[i] = 0 if 'dog' in str(image_filename) else 1
        i += 1   


# In[43]:

make_train_set(images_dir)


# In[45]:

from sklearn.cross_validation import train_test_split
X, X_test, Y, Y_test = train_test_split(all_X, all_y, test_size=0.1, random_state=42)


# In[52]:

import tflearn
from tflearn.data_utils import shuffle, to_categorical
# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)


# In[55]:

from tflearn.data_preprocessing import ImagePreprocessing
# normalisation of images
img_prepocessing = ImagePreprocessing()
img_prepocessing.add_featurewise_zero_center()
img_prepocessing.add_featurewise_stdnorm()


# In[60]:

# Input is a 64x64 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prepocessing)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_10.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')


# In[67]:

###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=1, run_id='model_cat_dog_10', show_metric=True)


model.save('model')
print("Model Saved")

model.load('modelo')
print("Modelo loaded")
# In[ ]:




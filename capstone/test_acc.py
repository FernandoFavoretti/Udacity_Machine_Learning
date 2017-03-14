#using the same format as the training
import numpy as np
import pandas as pd
from PIL import Image
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy
from glob import glob
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
from tflearn.data_utils import shuffle, to_categorical
import argparse



parser = argparse.ArgumentParser(description='Decide if a image is a dog or a cat')
parser.add_argument('dir', type=str, help='The dir of the test')
args = parser.parse_args()

# Same image preprocessing
img_size = 64

def transform_images(img, width, height):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    return img


img_prepocessing = ImagePreprocessing()
img_prepocessing.add_featurewise_zero_center()
img_prepocessing.add_featurewise_stdnorm()


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
model.load("model_cat_dog_10.tflearn-450")



image_filenames = os.listdir(args.dir)
nm_correct = 0
nm_wrong = 0

for image in image_filenames:
	image_path = os.path.join(args.dir, image)
	img = cv2.imread(image_path)
	img = transform_images(img, img_size, img_size)
	img = scipy.misc.imresize(img, (64, 64), interp="bicubic").astype(np.float32, casting='unsafe')
	#Predict
	prediction = model.predict([img])
	if prediction[0][0] > prediction [0][1] and 'dog' in str(image_path):
		nm_correct += 1
	elif prediction[0][1] > prediction [0][0] and 'cat' in str(image_path):	
		nm_correct += 1
	else:
		nm_wrong += 1
			
acc = float(nm_correct) / float(nm_correct+nm_wrong)*1.0			
print "Final Accuracy: "+ str(acc)









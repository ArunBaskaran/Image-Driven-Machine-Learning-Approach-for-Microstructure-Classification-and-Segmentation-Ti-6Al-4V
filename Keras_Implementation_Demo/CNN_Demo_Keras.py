

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import exposure
from skimage import feature
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras import regularizers


import os.path

def smooth(img):
    return 0.5*img + 0.5*(
        np.roll(img, +1, axis=0) + np.roll(img, -1, axis=0) +
        np.roll(img, +1, axis=1) + np.roll(img, -1, axis=1) )
    





xavier_init = tf.contrib.layers.xavier_initializer()  #Initializer for weights
zero_init = tf.zeros_initializer()  #Initializer for biases

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D( 2, [5,5], (1,1), input_shape = (200,200,1), kernel_initializer = xavier_init, bias_initializer = zero_init, kernel_regularizer=regularizers.l1(0.001), padding = 'valid', name = 'C1'),  
    keras.layers.MaxPool2D((2,2), (2,2), input_shape = (196,196,2),padding = 'valid', name ='P1'),
    keras.layers.Conv2D(4, [5,5],(1,1), input_shape = (98,98,2), kernel_initializer = xavier_init, bias_initializer = zero_init, kernel_regularizer=regularizers.l1(0.001), name ='C2'),  
    keras.layers.MaxPool2D((2,2), (2,2), input_shape = (94,94,4), padding = 'valid', name ='P2'),
    keras.layers.Conv2D(12, [3,3],(1,1), input_shape = (47,47,4), kernel_initializer = xavier_init, bias_initializer = zero_init, kernel_regularizer=regularizers.l1(0.001), name ='C3'),  
    keras.layers.Flatten(name ='fc_layer'),
    keras.layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l1(0.001)),
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model


df = pd.read_excel('labels.xlsx', header=None, names=['id', 'label'])
total_labels = df['label']
for i in range(len(total_labels)):
	total_labels[i]-=1



import random

width=200
height=200
total_size = 1225
train_size = 1000
validation_size = 100
test_size = total_size - train_size - validation_size

import random
train_list = random.sample(range(1,total_size+1), train_size)


nontrainlist = []
test_list = []
for i in range(1,total_size+1):
    if i not in train_list:
        nontrainlist.append(i)

validation_list = random.sample(nontrainlist, validation_size)

for item in nontrainlist:
    if(item not in validation_list):
        test_list.append(item)



print(len(nontrainlist))

print(len(train_list), len(validation_list), len(test_list))


# In[7]:


train_images = []
train_labels = []
validation_images = []
validation_labels = []
test_images = []
test_labels=[]


for i in range(1, total_size+1):
    if i in train_list:
        filename = 'image_' + str(i) + '.png'
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.blur(image,(5,5))
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        train_images.append(image)
        train_labels.append(total_labels[i-1])
    elif i in validation_list:
        filename = 'image_' + str(i) + '.png'
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.blur(image,(5,5))
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        validation_images.append(image)
        validation_labels.append(total_labels[i-1])
    else:
        filename = 'image_' + str(i) + '.png'
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.blur(image,(5,5))
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        test_images.append(image)
        test_labels.append(total_labels[i-1])



train_images = np.reshape(train_images, (train_size, width, height, 1))
validation_images = np.reshape(validation_images, (validation_size, width, height, 1))
test_images = np.reshape(test_images, (test_size, width, height, 1))

train_labels = tf.keras.backend.one_hot(train_labels,3)
test_labels = tf.keras.backend.one_hot(test_labels,3)
validation_labels = tf.keras.backend.one_hot(validation_labels,3)



#------------Training---------------#


model = create_model()

checkpoint_path = "weights/classification.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)

model.fit(train_images, train_labels, batch_size=200,  epochs=500, validation_data=(validation_images,validation_labels), steps_per_epoch = 1, validation_steps=1,
          callbacks=[cp_callback])  
          


#------------Testing---------------#

model = create_model()


model.load_weights(checkpoint_path)

loss,acc = model.evaluate(test_images,  test_labels, verbose=2, steps = 1)
print("Accuracy:  {:f}%".format(100*acc))


#----Verifying the distribution of predicted classes----#
y_prob = model.predict(test_images) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)


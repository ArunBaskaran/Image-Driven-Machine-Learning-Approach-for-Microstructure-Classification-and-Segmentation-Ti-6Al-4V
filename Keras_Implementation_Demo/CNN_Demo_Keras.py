"""
----------------------------------ABOUT-----------------------------------
Author: Arun Baskaran
--------------------------------------------------------------------------
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os.path

from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras import regularizers

from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max
from PIL import Image
from skimage import exposure, data, morphology
from skimage.color import label2rgb
from skimage.feature import hog
from skimage.filters import sobel



def smooth(img):
    return 0.5*img + 0.5*(
        np.roll(img, +1, axis=0) + np.roll(img, -1, axis=0) +
        np.roll(img, +1, axis=1) + np.roll(img, -1, axis=1) )
        
        
def returnIndex(a , value):
    k = np.size(a)
    for i in range(k):
        if(a[i]==value):
            return i
    




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
train_size = 900
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
test_images_id = []


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
        test_images_id.append(i)
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

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience = 50, mode='min', restore_best_weights=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)

model.fit(train_images, train_labels,  epochs=1500, validation_data=(validation_images,validation_labels), steps_per_epoch = 4, validation_steps=1, callbacks=[es, cp_callback])  

loss,acc = model.evaluate(test_images,  test_labels, verbose=2, steps = 1)
print("Accuracy:  {:5.2f}%".format(100*acc))
          


#------------Testing---------------#

model = create_model()


model.load_weights(checkpoint_path)

loss,acc = model.evaluate(test_images,  test_labels, verbose=2, steps = 1)
print("Accuracy:  {:f}%".format(100*acc))


#----Verifying the distribution of predicted classes----#
y_prob = model.predict(test_images) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)

#----Label-specific segmentation algorithm-----#

equiaxed_area_fraction_dict = {}
lamellae_area_fraction_dict= {}

for i in range(np.size(y_classes)):
    if(y_classes[i]==0):
        area_frac_duplex=[]
        duplex_image_id=[]
        filename = 'image_' + str(test_images_id[i]) + '.png'
        image = Image.open(filename).convert('F')
        image = np.copy(np.reshape(np.array(image), image.size[::-1])/255.)   
        image = exposure.equalize_adapthist(image, clip_limit=8.3)
        image = (smooth(smooth(image)))
        image_copy = image
        image = cv2.resize(image, dsize=(200,200), interpolation=cv2.INTER_CUBIC)
        image_copy = cv2.resize(image_copy, dsize=(200,200), interpolation=cv2.INTER_CUBIC)
        markers = np.zeros_like(image)
        markers[image > np.median(image) - 0.10*np.std(image)] = 1     
        markers[image < np.median(image) - 0.10*np.std(image)] = 2
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
        elevation_map = sobel(image)
        #The following implementation of watershed segmentation has been adopted from scikit's documentation example: https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
        segmentation = morphology.watershed(elevation_map, markers)
        segmentation = ndi.binary_fill_holes(segmentation - 1)
        labeled_grains, _ = ndi.label(segmentation)
        image_label_overlay = label2rgb(labeled_grains, image=image)
        ax1.imshow(image_copy, cmap=plt.cm.gray, interpolation='nearest')
        ax1.contour(segmentation, [0.5], linewidths=1.2, colors='r')
        ax1.axis('off')
        outfile = 'seg_duplex_' + str(test_images_id[i]) + '.png'
        plt.savefig(outfile, dpi=100)
        equiaxed_area_fraction_dict[test_images_id[i]] = np.sum(segmentation)/(np.shape(image)[0]*np.shape(image)[1])
    elif(y_classes[i]==1):
        dim = 400
        filename = 'image_' + str(test_images_id[i]) + '.png'
        image = Image.open(filename).convert('F')
        image = np.copy(np.reshape(np.array(image), image.size[::-1])/255.)
        image = exposure.equalize_hist(image)
        image = smooth(image)
        image = np.reshape(image, (np.shape(image)[0],np.shape(image)[1]))
        gx = cv2.Sobel(np.float32(image), cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(np.float32(image), cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag_cut_off = 0.2*np.max(mag)
        (n,bins,patches) = plt.hist(angle.ravel(), bins = 30)
        n_sorted = sorted(n, reverse=True)
        bin0 = bins[returnIndex(n, n_sorted[0])]
        bin1 = bins[returnIndex(n, n_sorted[1])]
        bin2 = bins[returnIndex(n, n_sorted[2])]
        bin_s = np.ones(20)
        for i in range(20):
            bin_s[i] = bins[returnIndex(n, n_sorted[i])]
        markers = np.zeros_like(angle)
        markers[(angle/360 > bin1/360 - 26/360) & (angle/360 < bin1/360 + 26/360) & (mag > mag_cut_off)] = 1      
        markers[(angle/360 > bin2/360 - 18/360) & (angle/360 < bin2/360 + 18/360) & (mag > mag_cut_off)] = 1  
        markers[(angle/360 > bin0/360 - 18/360) & (angle/360 < bin0/360 + 18/360) & (mag > mag_cut_off)] = 1 
        markers = (smooth(smooth(markers)))
        markers1 = np.where(markers > np.mean(markers), 1.0, 0.0)
        lamellae_area_fraction_dict[test_images_id[i]] = np.sum(markers1)/(np.shape(image)[0]*np.shape(image)[1])
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
        ax1.imshow(image, 'gray')
        ax1.imshow(markers1, alpha = 0.5)
        image1 = image + markers1
        ax1.imshow(image1)
        #plt.colorbar()
        outfile = 'seg_lamellae_' + str(test_images_id[i]) + '.png'
        plt.savefig(outfile, dpi=100)


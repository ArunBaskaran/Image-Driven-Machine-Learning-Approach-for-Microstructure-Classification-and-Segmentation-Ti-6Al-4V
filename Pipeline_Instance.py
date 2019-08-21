from IPython.display import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import numpy as np
import pandas as pd
from skimage import data, color, io, img_as_float, exposure, measure, morphology, feature
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max
from PIL import Image
from skimage.color import label2rgb
from skimage.feature import hog
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.filters import sobel
import random

final_width = 45 #Dimension of the final convolution layer is final_width x final_width


def meansmoothing(image):
    return 0.5*image + 0.5*(np.roll(image, +1, axis=0) + np.roll(image, -1, axis=0) + np.roll(image, +1, axis=1) + np.roll(image, -1, axis=1))

class Data:
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.batch_num = 0
    
    def full_batch(self):
        return np.array(self.X_data), np.array(self.Y_data)
    
    def random_batch(self, batch_size): 
        rand_batch_num = np.random.randint((len(self.X_data) / batch_size))
        X_batch = self.X_data[rand_batch_num*batch_size:(rand_batch_num+1)*batch_size]
        Y_batch = self.Y_data[rand_batch_num*batch_size:(rand_batch_num+1)*batch_size]
        return np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)


def localIntensity(image):
    image_new = np.zeros(np.shape(image))
    for i in range(3,np.shape(image)[0]-3):
        for j in range(3,np.shape(image)[1]-3):
            kernel = image[i-3:i+3, j-3:j+3]
            image_new[i][j] = (np.median(kernel))
    return image_new
    
def meansmoothing(img):
    return 0.5*img + 0.5*(
        np.roll(img, +1, axis=0) + np.roll(img, -1, axis=0) +
        np.roll(img, +1, axis=1) + np.roll(img, -1, axis=1) )


def convolutional_layer(input_layer, weights, biases, name):
    with tf.name_scope('conv_layer') as scope:
        conv_layer = tf.nn.conv2d(input_layer, weights, strides=[1,1,1,1], padding='VALID')
        conv_layer = tf.add(conv_layer,biases, name='{}_pre_act'.format(name))
        conv_layer = tf.nn.relu(conv_layer, name=name)
        return conv_layer


def output_layer(fc_layer, weights, biases):
    logits = tf.add(tf.matmul(fc_layer, weights), biases, name='logits')
    Y_hat = tf.nn.softmax(logits, axis=1, name='Y_hat')
    return logits, Y_hat


df = pd.read_excel('labels.xlsx', header=None, names=['id', 'label'])
total_labels = df['label']
for i in range(len(total_labels)):
	total_labels[i]-=1
print(np.shape(total_labels))


classes = 3
width = 200
height = 200
channels = 1

total_size = 1225 
train_size = 1000 
test_size = total_size-train_size


random_list = random.sample(range(1,1226), train_size)

train_images = []
test_images = []
train_labels=[]
test_labels=[]
train_image_id=[]
test_image_id=[]

df = pd.read_excel('labels.xlsx', header=None, names=['id', 'label'])
total_labels = df['label']
for i in range(len(total_labels)):
	total_labels[i]-=1
	
for i in range(1226):
    if i in random_list:
        filename = 'image_' + str(i) + '.png'
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.blur(image,(5,5))
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        train_image_id.append(i)
        train_images.append(image)
        train_labels.append(total_labels[i])
    else:
        filename = 'image_' + str(i) + '.png'
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.blur(image,(5,5))
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        test_image_id.append(i)
        test_images.append(image)
        test_labels.append(total_labels[i])
print(np.shape(train_images))
print(np.shape(test_images))

train_data = Data(train_images, np.array(train_labels))
test_data = Data(test_images, np.array(test_labels))

lr = .001
epochs = 600
batch_size = 200


tf.reset_default_graph()

xavier_init = tf.contrib.layers.xavier_initializer()
zero_init = tf.zeros_initializer()

W_c_1 = tf.get_variable(shape=[5, 5, channels,2], dtype=tf.float32, initializer=xavier_init, name='C1_weights')
W_c_2 = tf.get_variable(shape=[5, 5, 2, 4], dtype=tf.float32,initializer=xavier_init,name='C2_weights')
W_c_3 = tf.get_variable(shape=[3, 3, 4, 12], dtype=tf.float32,initializer=xavier_init,name='C3_weights')
W_fc = tf.get_variable(shape=[final_width*final_width*12, classes],dtype=tf.float32,initializer=xavier_init,name='fc_weights')

B_c_1 = tf.get_variable(shape=[2], dtype=tf.float32, initializer=zero_init, name='C1_biases')
B_c_2 = tf.get_variable(shape=[4], dtype=tf.float32, initializer=zero_init, name='C2_biases') 
B_c_3 = tf.get_variable(shape=[12], dtype=tf.float32, initializer=zero_init, name='C3_biases')
B_fc = tf.get_variable(shape=[classes], dtype=tf.float32, initializer=zero_init, name='fc_biases') 


image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channels], name='image')
label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
C1 = convolutional_layer(image, W_c_1, B_c_1, 'C1')
P1 = tf.nn.max_pool(C1, ksize=[1,2,2,1], padding='VALID', strides=[1,2,2,1], name='P1') #pooling_layer(C1, 2, 2, 'P1')
C2 = convolutional_layer(P1, W_c_2, B_c_2, 'C2')
P2 = tf.nn.max_pool(C2, ksize=[1,2,2,1], padding='VALID', strides=[1,2,2,1], name='P2') #pooling_layer(C2, 2, 2, 'P2')
C3 = convolutional_layer(P2, W_c_3, B_c_3, 'C3')
FC = tf.reshape(C3, shape=[-1, final_width*final_width*12], name='fc_layer')  #fc_layer(C3, 'fc_layer') 


logits = tf.add(tf.matmul(FC, W_fc), B_fc, name='logits')
Y_hat = tf.nn.softmax(logits, axis=1, name='Y_hat')


loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
loss = tf.reduce_mean(loss, name='binary_crossentropy_loss')
    
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(loss)

predict_op = tf.argmax(Y_hat, 1)
tf.get_collection('validation_nodes')
tf.add_to_collection('validation_nodes', image)
tf.add_to_collection('validation_nodes', label)
tf.add_to_collection('validation_nodes', predict_op)
saver = tf.train.Saver()


num_correct = tf.equal(label, tf.cast(tf.argmax(Y_hat, 1), tf.int32),name='num_correct')
accuracy = tf.reduce_mean(tf.cast(num_correct, dtype=tf.float32),name='accuracy')

plot_loss_train = []

hist_train = []
plot_loss_test = []
hist_test = []

test_acc_ops = {'X':image,'Y':label,'loss':loss,'accuracy_op':accuracy}
confusion_matrix_op = tf.confusion_matrix(label, predict_op, num_classes=classes)

#--------------------------------Training the neural network-----------------------------------------#
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    
    for epoch in range(1,epochs+1):
        X_batch_train, Y_batch_train = train_data.random_batch(batch_size)
        X_batch_train = np.reshape(X_batch_train, (batch_size, width, height, channels))
        session.run([train], feed_dict={image: X_batch_train, label: Y_batch_train})

        
        if epoch%100 == 0:
            train_loss, train_acc = session.run([loss, accuracy], feed_dict={image: X_batch_train, label: Y_batch_train})
            plot_loss_train.append(train_loss)
            hist_train.append(train_acc)
        if epoch%100 == 0:
            X_batch_test, Y_batch_test = test_data.full_batch()
            X_batch_test = np.reshape(X_batch_test, (test_size, width, height, channels))
            test_loss, test_acc = session.run([loss, accuracy], feed_dict={image: X_batch_test, label: Y_batch_test})
            plot_loss_test.append(test_loss)
            hist_test.append(test_acc)
            
        if(epoch%100 == 0):
            print('Epoch: {:<10} | Loss: {:<25} | Test Accuracy {:<20}'.format(epoch, test_loss, test_acc))


    save_path = saver.save(session, "./classification") 
    confusion_matrix = []
    X_batch_test, Y_batch_test = test_data.full_batch()
    X_batch_test = np.reshape(X_batch_test, (test_size, 200,200,1))
    predictions = predict_op.eval(feed_dict = {image: X_batch_test})
    matrix = session.run(confusion_matrix_op, feed_dict={image:X_batch_test, label:Y_batch_test})
    confusion_matrix.append(matrix)
    confusion_matrix = sum(confusion_matrix)   
    
#------------------------------------------------------------------------------------------------------#


#-------------------Visualizing the classification step through a confusion matrix---------------------#
    
categories = ['lamellar', 'duplex', 'martensite']
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(confusion_matrix)

ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(categories)))
ax.set_xticklabels(categories)
ax.set_yticklabels(categories)
plt.rc('xtick', labelsize=14)    
plt.rc('ytick', labelsize=14)    

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(categories)):
    for j in range(len(categories)):
        text = ax.text(j, i, confusion_matrix[i][j], ha="center", va="center", color="w")

ax.set_title("Confusion Matrix",fontsize=20)
fig.tight_layout()
plt.savefig('confusion.png')

#-------------------------------------------------------------------------------------------------------#




#---------------------Segmenting Microstructures---------------------------------------------------------#

equiaxed_area_fraction_dict = {}
lamellae_area_fraction_dict= {}

for i in range(np.size(predictions)):
	if(predictions[i]==0):
		area_frac_duplex=[]
        duplex_image_id=[]
        filename = 'image_' + str(test_image_id[i]) + '.png'
        image = Image.open(filename).convert('F')
        image = np.copy(np.reshape(np.array(image.getdata()), image.size[::-1])/255.)
        image_copy = image
        image = exposure.equalize_adapthist(image, clip_limit=8.3)
        image = (meansmoothing(meansmoothing(image)))
        image = cv2.resize(image, dsize=(200,200), interpolation=cv2.INTER_CUBIC)
        image_copy = cv2.resize(image_copy, dsize=(200,200), interpolation=cv2.INTER_CUBIC)
        image_new = (meansmoothing(meansmoothing(image)))
        markers1 = np.zeros_like(image_new)
        markers1[image_new > np.median(image_new) - 0.10*np.std(image_new)] = 1     
        markers1[image_new < np.median(image_new) - 0.10*np.std(image_new)] = 2
        fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
        elevation_map = sobel(image_new)
        segmentation1 = morphology.watershed(elevation_map, markers1)
        segmentation1 = ndi.binary_fill_holes(segmentation1 - 1)
        labeled_grains, _ = ndi.label(segmentation1)
        image_label_overlay = label2rgb(labeled_grains, image=image)
        ax1.imshow(image_copy, cmap=plt.cm.gray, interpolation='nearest')
        ax1.contour(segmentation1, [0.5], linewidths=1.2, colors='r')
        ax1.axis('off')
        outfile = 'seg_duplex_' + str(test_image_id[i]) + '.png'
        plt.savefig(outfile, dpi=100)
        equiaxed_area_fraction_dict[test_image_id[i]] = np.sum(segmentation1)/(np.shape(image)[0]*np.shape(image)[1])
        
    elif(prediction[i]==1):
		dim = 400
		filename = 'image_' + str(test_image_id[i]) + '.png'
		image = Image.open(filename).convert('F')
        image = np.copy(np.reshape(np.array(image.getdata()), image.size[::-1])/255.)
        image1 = Image.open(filename).convert('F')
        image1 = np.copy(np.reshape(np.array(image1.getdata()), image1.size[::-1])/255.)
        image = exposure.equalize_hist(image)
        image = meansmoothing(image)
        image2 = np.reshape(image, (np.shape(image)[0],np.shape(image)[1]))
        gx = cv2.Sobel(np.float32(image2), cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(np.float32(image2), cv2.CV_32F, 0, 1, ksize=1)
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
        markers2 = np.zeros_like(angle)
        markers2[(angle/360 > bin1/360 - 26/360) & (angle/360 < bin1/360 + 26/360) & (mag > mag_cut_off)] = 1      
        markers2[(angle/360 > bin2/360 - 18/360) & (angle/360 < bin2/360 + 18/360) & (mag > mag_cut_off)] = 1  
        markers2[(angle/360 > bin0/360 - 18/360) & (angle/360 < bin0/360 + 18/360) & (mag > mag_cut_off)] = 1 
        markers2 = (meansmoothing(meansmoothing(markers2)))
        markers3 = np.where(markers2 > np.mean(markers2), 1.0, 0.0)
        lamellae_area_fraction_dict[test_image_id[i]] = np.sum(markers3)/(np.shape(image)[0]*np.shape(image)[1])
		fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
        ax1.imshow(image, 'gray')
        ax1.imshow(markers2, alpha = 0.5)
        image3 = image + markers2
        ax1.imshow(image3)
        plt.colorbar()
        outfile = 'seg_lamellae_' + str(test_image_id[i]) + '.png'
        plt.savefig(outfile, dpi=100)

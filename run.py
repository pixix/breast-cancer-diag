#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:58:06 2018

@author: xixiong
"""

#import config
import os
import glob
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras_resnet.models
#from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras import callbacks
import os
import glob
from sklearn.model_selection import train_test_split
from keras import backend as K
from livelossplot import PlotLossesKeras
from skimage.measure import block_reduce
from keras.utils import multi_gpu_model

K.image_data_format()
K.tensorflow_backend._get_available_gpus()
#%%
#filename = '/Users/xixiong/Documents/Dataset/breastcancer/Mammo_Image_Positive/201806080009/LCC'
#filename = os.path.join(filename, '1.3.6.1.4.1.30071.8.10769794.5817621274657144.dcm')
#ds = pydicom.dcmread(filename)
#DATA_PATH = '/Mammo_Image'
#DATA_PATH = '/home/ubuntu/Mammo_Image'
DATA_PATH = '/Users/xixiong/Documents/Dataset/breastcancer/'
#DATA_PATH ='/Users/xixiong/Documents/Dataset/breastcancer/Mammo_Image_Positive/201806080012/RCC'
k1=0
k2=0
dataset = [np.ndarray(shape =(895, 703))]


#for file in glob.glob(os.path.join(DATA_PATH,'*/**/**.raw'),recursive=True):
#    print (file)
#    os.remove(file)


for file in glob.glob(os.path.join(DATA_PATH,'Mammo_Image_Positive/**/**.dcm'),recursive=True):
    ds = pydicom.dcmread(file)
    print (file)
    #plt.imshow(ds.pixel_array,cmap = plt.cm.bone)
    #plt.imsave(file.split('.dcm')[0]+'1.png',ds.pixel_array)
    #plt.show()
    dataset = np.append(dataset,[block_reduce(ds.pixel_array,(4,4),np.mean)],axis=0)
    k1 += 1
    #if k1 == 3:break
for file in glob.glob(os.path.join(DATA_PATH,'Mammo_Image_Negative/**/**.dcm'),recursive=True):
    ds = pydicom.dcmread(file)
    print (file)
    dataset = np.append(dataset,[block_reduce(ds.pixel_array,(4,4),np.mean)],axis=0)
    k2 += 1
    if k1 == k2-20: break
y1 = np.array([[1] for _ in range(k1)])
y2 = np.array([[0] for _ in range(k2)])
y = np.vstack((y1,y2))
del y1,y2
dataset = dataset[1:]
#%%
batch_size = 8
num_classes = 2
epochs = 10
data_augmentation = True
num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')

x_train, x_test, y_train, y_test = train_test_split(dataset,y, train_size = 0.8, random_state = 42)


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_test /=16381
x_train /=16381

img_rows, img_cols = 895, 703 # 3580, 2812
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x = keras.layers.Input(input_shape)
model = keras_resnet.models.ResNet50(x, classes = num_classes)

def recall(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#model = multi_gpu_model(model, gpus = 8)

model.compile(loss = 'categorical_crossentropy',
			  optimizer = "adam",
			  metrics = ['accuracy',recall])



tensorboard = callbacks.TensorBoard(log_dir = 'logs/12', histogram_freq = 0, write_graph = True, write_images = True)
model.fit(x_train,y_train,
			batch_size = batch_size,
			epochs = epochs,
			validation_data = (x_test,y_test),
			shuffle = True, callbacks = [PlotLossesKeras()])


'''
model = Sequential()
model.add(AveragePooling2D(pool_size = (2,2), input_shape = input_shape))
model.add(Conv2D(32, (3,3), padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding = 'same',activation = 'relu'))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = 'softmax'))
model.summary()

opt = keras.optimizers.rmsprop(lr = 0.001, decay = 1e-6)

model.compile(loss = 'categorical_crossentropy',
			  optimizer = opt,
			  metrics = ['accuracy'])
#model.pop()
#model.summary()

tensorboard = callbacks.TensorBoard(log_dir = 'logs/1', histogram_freq = 0, write_graph = True, write_images = True)

model.fit(x_train,y_train,
			batch_size = batch_size,
			epochs = epochs,
			validation_data = (x_test,y_test),
			shuffle = True, callbacks = [tensorboard])

score, rec = model.evaluate(x_test, y_test, batch_size = batch_size)'''

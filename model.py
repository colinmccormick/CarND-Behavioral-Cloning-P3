# model.py
# training model for behavioral cloning
# udacity nano-degree in self-driving car engineering

# import packages

import csv
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Cropping2D, ELU
from keras.callbacks import EarlyStopping

# set params
offset_angle = 0.20
test_sample_fraction = 0.20
im_shape = (160,320,3)
nvidia_crop = ((54,40),(60,60))

cameras = { 'center':{'index':0, 'offset':0},
            'left':{'index':1, 'offset': + offset_angle},
            'right':{'index':2, 'offset': - offset_angle} }

# data location
im_folder = "./data/IMG/"
csv_file = "./data/driving_log.csv"

# read driving log frame-by-frame
samples_raw = []
with open(csv_file,'r') as f:
    datareader = csv.reader(f)
    for line in datareader:
        samples_raw.append(line)
        
train_samples, valid_samples = train_test_split(samples_raw, test_size=test_sample_fraction)
print("Train samples: {0}; validation samples: {1}".format(len(train_samples),len(valid_samples)))

# define generator for training data
# include left/right cameras
# include augmentation (horiztonal flipping)
# include small-angle steering supression
# include brightness randomization

def generator_train(samples, batch_size=128):
    # generator for batches of training images
    
    while True:
        steering_angles = []
        images = []
        
        while len(images) < batch_size:
            
            # pick random sample and get steering angle
            sample = random.choice(samples)    
            steering_angle = float(sample[3]) 
            
            # choose which camera
            cam = cameras[random.choice(list(cameras))]
            steering_angle += cam['offset']
            
            # choose possible inversion/flip for data augmentation
            invert = random.choice([True,False])
            if invert:
                steering_angle = - steering_angle
            steering_angles.append(steering_angle)
        
            # get image
            im_file = sample[cam['index']].split('/')[-1]
            im = cv2.imread(im_folder + im_file)       
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if invert:
                im = cv2.flip(im,+1) 
            images.append(im)

        yield shuffle(np.array(images),np.array(steering_angles))

train_generator = generator_train(train_samples)

# define generator for validation data
# include left/right cameras

def generator_valid(samples, batch_size=128):
    # generator for batches of validation images
    
    while True:
        steering_angles = []
        images = []
        
        while len(images) < batch_size:
            
            # pick random sample and get steering angle
            sample = random.choice(samples)    
            steering_angle = float(sample[3])
            
            # choose which camera
            cam = cameras[random.choice(list(cameras))]
            steering_angle += cam['offset']
            steering_angles.append(steering_angle)
        
            # get image
            im_file = sample[cam['index']].split('/')[-1]
            im = cv2.imread(im_folder + im_file)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            images.append(im)
                
        yield shuffle(np.array(images),np.array(steering_angles))
        
valid_generator = generator_valid(valid_samples)

# build model - Nvidia

def model_nvidia():
    
    model = Sequential()
    model.add(Cropping2D(nvidia_crop,input_shape=im_shape))
    model.add(Lambda(lambda x: x/255 - 0.5))

    # convolutional layers 
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,3,3,activation='elu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,3,3,activation='elu'))
    model.add(Dropout(0.25))

    # fully connected
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model
    
# compile model
model = model_nvidia()
model.compile(loss='mse',optimizer='adam')

# fit model
# annoying but unharmful bug in fit_generator is documented here: 
# https://stackoverflow.com/questions/41789961/keras-warning-epoch-comprised-more-than-samples-per-epoch-samples

early_stop = EarlyStopping(min_delta = 0.00001, patience = 32)
history_object = model.fit_generator(train_generator,
                                    samples_per_epoch=len(train_samples),
                                    validation_data=valid_generator,
                                    nb_val_samples=len(valid_samples),
                                    callbacks = [early_stop],
                                    nb_epoch=256)

# save model
model.save('model.h5')


#!python3

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import *

REL_PATH = '../'
CSV_NAME = 'driving_log_udacity.csv'


def df_shuffle(df):
    return df.sample(frac=1).reset_index(drop=True)


def get_data():
    
    #parameters for adjusting steering angles for left and right images
    steering_add = 0.27
    steering_mult = 1.65

    data = pd.read_csv(REL_PATH + CSV_NAME)

    # get center images
    data_center = data[['center', 'steering']]
    data_center = data_center.rename(columns={'center': 'src'})

    # get left images and adjust the steering angle
    data_left = data[['left', 'steering']].copy()
    data_left.steering = data_left.steering * steering_mult + steering_add
    data_left = data_left.rename(columns={'left': 'src'})

    # get right images and adjust the steering angle
    data_right = data[['right', 'steering']].copy()
    data_right.steering = data_right.steering * steering_mult - steering_add
    data_right = data_right.rename(columns={'right': 'src'})

    # concatinate data for center, left and right images
    data = pd.concat([data_center, data_left, data_right], ignore_index=True)
    
    # add flipping flag
    data_flipped = data.copy()
    data['flip'] = 0
    data_flipped['flip'] = 1

    # concatinate and shuffle
    all_data = pd.concat([data, data_flipped], ignore_index=True)
    all_data = df_shuffle(all_data)
    return all_data


def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=(160, 320, 3)))
    model.add(AveragePooling2D((5, 5), border_mode='valid')) # resize the image
    model.add(Lambda(lambda x: x/127.5 - 1.)) # normalize
    model.add(Conv2D(3, 1, 1, border_mode='same')) # color space covnerter
    model.add(Conv2D(4, 3, 3, border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    model.add(Dropout(0.25))
    model.add(Conv2D(4, 3, 3, border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def images_generator(data, batch_size = 128):
    num_data = len(data)

    while 1:
        for offset in range(0, num_data, batch_size):
            batch_images = []
            batch_angles = []
            end = min(offset + batch_size, num_data)
            for idx in range(offset, end):
                # load the image and get the angle
                d = data.iloc[idx]
                image = mpimg.imread(REL_PATH + d.src.strip())
                angle = d.steering

                # flip the image and angle
                if d.flip == 1:
                    image = np.fliplr(image)
                    angle = -angle

                # randomly adjust brightness
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
                image[:,:,2] = image[:,:,2] * np.random.uniform(0.3, 1.3)
                image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB) 

                batch_images.append(image)
                batch_angles.append(angle)
            
            yield np.array(batch_images), np.array(batch_angles)


data = get_data()

train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
print("Number of training examples =", len(train_data))
print("Number of validation examples =", len(valid_data))

model = get_model()

model.fit_generator(images_generator(train_data), samples_per_epoch=len(train_data), nb_epoch=10, 
                    validation_data=images_generator(valid_data), nb_val_samples=len(valid_data)) 


print("Saving the model to disk")

model.save_weights("model.h5")

json_string = model.to_json()
with open("model.json", "w") as file:
    file.write(json_string)

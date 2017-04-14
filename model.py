import os
import argparse

import numpy as np
import pandas as pd
import cv2
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CROP_TOP, CROP_BOTTOM = 60, 26
random.seed()

def create_model(input_shape, model_creator):
    model = Sequential()

    model.add(Cropping2D(((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255) - 0.5))

    model_creator(model)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )
    return model

def my_model(model):
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(96, (3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(96, (3, 3), padding='valid', activation='elu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1, trainable=False))

def nvidia_model(model):
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.1))

    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, trainable=False))


def train(model, data, batch_size=32, epochs=5):
    data_train, data_test = train_test_split(data, test_size=0.3)
    steps_per_epoch = len(data_train) // batch_size
    print('Training set size:', len(data_train))
    print('Epochs:', epochs)
    print('Steps per epoch:', steps_per_epoch)
    model.fit_generator(
        generator(data_train, batch_size),
        steps_per_epoch,
        epochs=epochs
    )

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_with_canny(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    low_threshold = 50
    high_threshold = 150
    return cv2.Canny(blurred, low_threshold, high_threshold)

def adjust_brightness(img_steering, factor):
    img, steering = img_steering
    result = img * (1 + factor)
    return normalize(result), steering

def flip(img, steering):
    return np.fliplr(img), -steering

def normalize(img):
    return (img - img.mean()) / (img.max() - img.min())

def get_side_image(left: bool, image, center_steering, steering_correction=0.2):
    steering = (center_steering + steering_correction) if left else (center_steering - steering_correction)
    return image, steering

def augment(row, center_steering):
    images = []
    steerings = []

    def add(i, s):
        images.append(i)
        steerings.append(s)

    factor = random.uniform(0.1, 0.4)

    center = load_image(row['center'])

    left, left_steering = get_side_image(True, load_image(row['left']), center_steering)
    right, right_steering = get_side_image(True, load_image(row['right']), center_steering)

    for image, steering in ((center, center_steering), (left, left_steering), (right, right_steering)):
        add(image, steering)
        add(*adjust_brightness((image, steering), factor))
        add(*adjust_brightness((image, steering), -factor))

    add(*flip(center, center_steering))
    add(*flip(*adjust_brightness((center, center_steering), factor)))
    add(*flip(*adjust_brightness((center, center_steering), -factor)))

    return images, steerings

def generator(data, batch_size=32):
    size = len(data)
    while True:
        shuffle(data)
        for offset in range(0, size, batch_size):
            batch = data.iloc[offset:offset+batch_size]

            images = []
            steerings = []
            for i in range(len(batch)):
                row = batch.iloc[i]
                steering = row['steering']
                imgs, steers = augment(row, steering)
                for img, steer in zip(imgs, steers):
                    images.append(img)
                    steerings.append(steer)
            yield shuffle(np.array(images), np.array(steerings))

def read_csv(filename):
    columns = ( 'center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed' )
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = columns

    return df

def fix_image_paths(df, folder):
    df['center'] = df['center'].map(lambda s: os.path.join(folder, s.split('/')[-1]))
    df['left'] = df['left'].map(lambda s: os.path.join(folder, s.split('/')[-1]))
    df['right'] = df['right'].map(lambda s: os.path.join(folder, s.split('/')[-1]))
    return df

def detect_input_shape(df):
    sample_image = df['center'].loc[1]
    return load_image(sample_image).shape

def parse_args():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='recordings',
        help='Path to CSV and image folder.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = read_csv(os.path.join(args.image_folder, 'driving_log.csv'))
    data = fix_image_paths(data, os.path.join(args.image_folder, 'IMG'))
    input_shape = detect_input_shape(data)

    model = create_model(input_shape, my_model)
    print(model.summary())

    train(model, data, batch_size=64, epochs=4)

    model.save(args.model)


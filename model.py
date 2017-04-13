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


def train(model, x, y, batch_size=32, epochs=5):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    steps_per_epoch = len(y_train) // batch_size
    print('Training set size:', len(y_train))
    print('Epochs:', epochs)
    print('Steps per epoch:', steps_per_epoch)
    model.fit_generator(
        generator(X_train, y_train, batch_size),
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

def adjust_brightness(img, factor):
    result = img * (1 + factor)
    return normalize(result)

def flip(img, steering):
    return np.fliplr(img), -steering

def normalize(img):
    return (img - img.mean()) / (img.max() - img.min())

def augment(image, steering):
    images = []
    steerings = []

    def add(i, s):
        images.append(i)
        steerings.append(s)

    factor = random.uniform(0.1, 0.4)
    add(image, steering)
    add(adjust_brightness(image, factor), steering)
    add(adjust_brightness(image, -factor), steering)

    add(*flip(image, steering))
    add(*flip(adjust_brightness(image, factor), steering))
    add(*flip(adjust_brightness(image, -factor), steering))

    return images, steerings

def generator(x, y, batch_size=32):
    size = len(y)
    while True:
        shuffle(x, y)
        for offset in range(0, size, batch_size):
            batch_x = x[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]

            images = []
            steerings = []
            for i, (filename, steering) in enumerate(zip(batch_x, batch_y)):
                imgs, steers = augment(load_image(filename), steering)
                for img, steer in zip(imgs, steers):
                    images.append(img)
                    steerings.append(steer)
            yield shuffle(np.array(images), np.array(steerings))

def read_csv(filename, steering_adjustment=0.2):
    columns = ( 'center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed' )
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = columns

    df['steering_left'] = df['steering'] + steering_adjustment
    df['steering_right'] = df['steering'] - steering_adjustment

    result_columns = ('image', 'steering')
    center = df[['center', 'steering']].as_matrix()
    left = df[['left', 'steering_left']].as_matrix()
    right = df[['right', 'steering_right']].as_matrix()
    return pd.DataFrame(data=np.concatenate((center, left, right)), columns=result_columns)

def fix_image_paths(df, folder):
    df['image'] = df['image'].map(lambda s: os.path.join(folder, s.split('/')[-1]))
    return df

def detect_input_shape(df):
    sample_image = df['image'].loc[1]
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

    data = read_csv(os.path.join(args.image_folder, 'driving_log.csv'), steering_adjustment=0.2)
    data = fix_image_paths(data, os.path.join(args.image_folder, 'IMG'))
    input_shape = detect_input_shape(data)

    model = create_model(input_shape, my_model)
    print(model.summary())

    train(model, data['image'], data['steering'], batch_size=64, epochs=5)

    model.save(args.model)


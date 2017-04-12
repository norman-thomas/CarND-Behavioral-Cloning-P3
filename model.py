import os
import argparse

import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CROP_TOP, CROP_BOTTOM = 60, 26
CHANNELS = 1

# exponential moving average
def ema(input_i, size):
    initial_weight = 1
    factor = 0.25
    weights = np.array([ initial_weight * (factor ** i) for i in range(size) ])
    values = input_i[:size]
    if len(values) < size:
        weights = weights[:len(values)]
    return np.dot(values, weights) / weights.sum()


def my_model(input_shape):
    model = Sequential()

    model.add(Conv2D(24, (3, 3), input_shape=input_shape))
    model.add(Activation('elu'))

    model.add(Conv2D(48, (3, 3)))
    model.add(Activation('elu'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(16))
    model.add(Activation('elu'))

    model.add(Dense(1))
    # TODO
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    return model

def train(model, x, y, batch_size=32, epochs=5):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    steps_per_epoch = 100 #len(y_train) // batch_size
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
    if CHANNELS == 1:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    low_threshold = 50
    high_threshold = 150
    return cv2.Canny(blurred, low_threshold, high_threshold)


def generator(x, y, batch_size=32):
    size = len(y)
    while True:
        shuffle(x, y)
        for offset in range(0, size, batch_size):
            batch_x = x[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]

            height = (160 - CROP_TOP - CROP_BOTTOM) // 2
            width = 160
            images = np.zeros((batch_size, height, width, CHANNELS))
            steerings = np.zeros((batch_size, 1))
            for i, (filename, steering) in enumerate(zip(batch_x, batch_y)):
                img = load_image(filename)
                img = img[CROP_TOP:160-CROP_BOTTOM,:]
                h, w = img.shape[:2]
                img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
                img = ((img / 255) - 0.5).astype(np.float16)
                images[i] = img if CHANNELS > 1 else img.reshape((height, width, 1))
                steerings[i] = steering
            yield shuffle(np.array(images), np.array(steerings))

def read_csv(filename, steering_adjustment=0.04):
    columns = ( 'center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed' )
    df = pd.read_csv(filename, names=columns)

    df['steering_left'] = df['steering'] + steering_adjustment
    df['steering_right'] = df['steering'] - steering_adjustment

    result_columns = ('image', 'steering')
    center = df[['center', 'steering']].as_matrix()
    left = df[['left', 'steering_left']].as_matrix()
    right = df[['right', 'steering_right']].as_matrix()

    return pd.DataFrame(data=np.concatenate((center, left, right)), columns=result_columns)


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
    data['image'] = data['image'].map(lambda s: os.path.join(args.image_folder, 'IMG', s.split('/')[-1]))
    model = my_model((80-30-13, 160, CHANNELS))
    print(model.summary())
    train(model, data['image'], data['steering'], batch_size=32)
    model.save(args.model)


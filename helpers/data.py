import os
import pandas as pd
import numpy as np
import cv2

# read a single CSV file into a pandas DataFrame
def read_csv(filename):
    columns = ('center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed')
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = columns
    return df

# Exponential Moving Average of steering angles
def smoothen_steering(df, count=5):
    return df.ewm(span=count).mean()['steering']

# rewrite image paths from absolute to relative paths as I occasionally moved files
def fix_image_paths(df, folder):
    fixpath = lambda s: os.path.join(folder, s.split('/')[-1])
    df['center'] = df['center'].map(fixpath)
    df['left'] = df['left'].map(fixpath)
    df['right'] = df['right'].map(fixpath)
    return df

# load image from file in HSV color space
def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# return the shape of the first image in data frame
def detect_input_shape(df):
    sample_image = df['center'].iloc[1]
    return load_image(sample_image).shape

# reads all driving_log.csv files in direct subfolders and concatenate data
def read_all_csvs_in_folder(parent_folder, min_speed=0.1):
    folders = os.listdir(parent_folder)
    single_folder = False
    if len(folders) == 2:
        items = set(folders)
        if 'driving_log.csv' in items and 'IMG' in items:
            folders = [parent_folder]
            single_folder = True

    if not single_folder:
        folders = [os.path.join(parent_folder, folder) for folder in folders]

    data = []
    for folder in folders:
        df = read_csv(os.path.join(folder, 'driving_log.csv'))
        df = fix_image_paths(df, os.path.join(folder, 'IMG'))
        df['smooth_steering'] = smoothen_steering(df, count=10)
        data.append(df)

    columns = data[0].columns
    data = pd.DataFrame(data=np.concatenate(data), columns=columns)
    return data[data['speed'] >= min_speed]


import os
import pandas as pd
import numpy as np
import cv2

def read_csv(filename):
    columns = ('center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed')
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = columns
    return df

def smoothen_steering(df, count=5):
    return df.ewm(span=count).mean()['steering']

def fix_image_paths(df, folder):
    fixpath = lambda s: os.path.join(folder, s.split('/')[-1])
    df['center'] = df['center'].map(fixpath)
    df['left'] = df['left'].map(fixpath)
    df['right'] = df['right'].map(fixpath)
    return df

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def detect_input_shape(df):
    sample_image = df['center'].iloc[1]
    return load_image(sample_image).shape

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


import random
import numpy as np
from .data import load_image

# change brightness of image by amount
def adjust_brightness(img_steering, amount):
    img, steering = img_steering
    result = img.copy().astype(np.int16)
    result[:,:,2] += amount
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, steering

# add random noise to an image with noise values ranging from 0 to maximum
def add_noise(img_steering, maximum=10):
    img, steering = img_steering
    noise = np.random.randint(0, high=maximum, size=img.shape)
    result = img.copy().astype(np.int16)
    result = np.clip(np.rint(result + noise), 0, 255).astype(np.uint8)
    return result, steering

# flip image hirozontally
def flip(img, steering):
    return np.fliplr(img.copy()), -steering

# calculate steering for left/right camera images
def get_side_image(left: bool, image, center_steering, steering_correction=0.25):
    steering = (center_steering + steering_correction) if left else (center_steering - steering_correction)
    return image, steering

# augment data by using left/right images and adding noise, changing brightness of center/left/right images
# and flipping center images
def augment(row, center_steering, threshold=0.05, droprate=0.7):
    images = []
    steerings = []

    def add(i, s):
        images.append(i)
        steerings.append(s)

    amount = int(random.uniform(20, 50))
    drop = random.uniform(0.0, 1.0)

    center = load_image(row['center'])

    left, left_steering = get_side_image(True, load_image(row['left']), center_steering)
    right, right_steering = get_side_image(False, load_image(row['right']), center_steering)

    work_on = [(left, left_steering), (right, right_steering)]
    if abs(center_steering) >= threshold or drop >= droprate:
        work_on.append((center, center_steering))
    for image, steering in work_on:
        add(image, steering)
        add(*adjust_brightness((image, steering), amount))
        add(*adjust_brightness((image, steering), -amount))
        add(*add_noise((image, steering)))

    add(*flip(center, center_steering))
    add(*flip(*adjust_brightness((center, center_steering), amount)))
    add(*flip(*adjust_brightness((center, center_steering), -amount)))
    add(*flip(*add_noise((center, center_steering))))

    return images, steerings


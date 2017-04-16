# CarND Project 3: Behavioral Cloning

## Writeup Template

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[steering_histogram]: ./model/output_5_1.png "Histogram of steering angles"
[sample_images]: ./model/output_11_1.png "Original sample images"
[sample_images_cropped]: ./model/output_12_1.png "Cropped sample images"
[sample_images_bright]: ./model/output_19_0.png "Brightened sample images"
[sample_images_dark]: ./model/output_20_0.png "Darkened sample images"
[sample_images_noise]: ./model/output_21_0.png "Noise sample images"
[sample_images_flipped]: ./model/output_22_0.png "Flipped sample images"


# Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/\#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

---

## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

| file | description |
|---|---|
| `model.py` | containing the script to create and train the model |
| `helpers/data.py` | helper module to handle data wrangling |
| `helpers/augment.py` | helper module to augment training data on the fly |
| `model.h5` | containing a trained convolution neural network |
| `drive.py` | for driving the car in autonomous mode |
| `video.mp4` | video recording of autonomous mode using my model `model.h5` on track 1 |
| `README.md` | project writeup |

While developing and training I used Python 3.5.2 with Keras 2.0.2 and TensorFlow 1.0.1.

### 2. Submission includes functional code

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Training Data

### 1. Collection

Due to a lack of confidence in my car simulator driving skills, which are based on experience with 3D racing games, I immediately opted for the training data provided by Udacity. Later on, I decided to practise my simulator driving skills and record some more data to extend the provided data set with some curvy driving and also with recovery driving.

### 2. Analysis

A first look at the data provided by Udacity reveals that in includes by far more straight-ahead driving (steering angle around 0.0) than left/right steering. That's when I decided to collect my own data additionally, as track 1 does not have that many straight sections. I drove on track 1 in both directions.

After adding my own recordings, the data now looks as follows:

| steering direction | count |
| --- | --- |
| **left** (angle < 0) | 3883 |
| **right** (angle > 0) | 3740 |
| **straight** (angle = 0) | 4420 |

There are 12,043 data points in total.

Here's a histogram of steering angles including both my own and Udacity's training data:

![Histogram of steering angles][steering_histogram]

There's still disproportionately more data for driving straight-ahead.

The input images have the size `160 x 320 x 3`. Here are 4 sample images:

![Original sample iamges][sample_images]

### 3. Preprocessing

#### Data Augmentation

Due to the high amount of data with `steering = 0` I decided to randomly drop that data from the training data. Additionally, in order to increase the overall amount of training data and make the model more robust and generalized, I decided the augment the data in several ways:

1. **change brightness** of input image
   * this helps the model generalize beyond brightness/darkness of a scene
2. **add random noise** to input image
   * this helps generate non-identical training images for the same steering angle
3. **flip** input image of center camera
   * this helps increase the amount of training data for left/right steering

Flipping is additionally applied to brightnened/darkened images as well as images with added noise.

Here are the various augmentations applied to the above sample images:


##### Brightened

![Brightened sample iamges][sample_images_bright]

##### Darkened

![Darkened sample iamges][sample_images_dark]

##### Added Noise

![Noise sample iamges][sample_images_noise]

##### Flipped

![Flipped sample iamges][sample_images_flipped]

#### Actual Preprocessing

I decided to pass the images into the model in HSV color space as this not only simplifies modification of brightness but also, to me, seems like a more 'semantic' format.

Besides transformation into HSV, all further preprocessing is done within the model.

The first layer of the model crops the image, thereby removing the part of the image above the horizon (i.e. sky) and the lower part including the hood of the car. I remove both because the hood is a constant, never changing part of the image, which provides no additional valueable information. For the same reason I also crop the sky as it keeps changing, but obviously does not determine the steering angle. Cropping also reduces the input size of the model, therefore reducing it's number of parameters and computation needed for training.

The second layer normalizes the input image of dimension `74 x 320 x 3` from integer values ranging between `0` and `255` to floating point values between `-0.5` up to `0.5`.



## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][steering_histogram]

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

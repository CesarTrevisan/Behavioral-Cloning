# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./center_2016_12_01_13_30_48_287.jpg "Center Camera"
[chart]: ./chart.jpg "Train and Validation Error"
[training_aws]: ./training_aws.jpg "Training Process"
[Nvidia]: ./neural_nvidia.png "Nvidia Architecture"


## Rubric Points

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
---

## Model Architecture and Training Strategy

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### 1. Model Architeture

In my final model I used [Nvidia's SelfDrive Car Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which is represented bellow:

![Alt text][Nvidia]

My model consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. (model.py lines 108-137) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 42). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

### 4. Appropriate training data

The Udacity's Simulator capture three images of each frame of video footage (Center, Left and Right cameras)

![Alt text][center]

And also capture steering angles and throttle, that are Lables.

## Solution Design Approach

First I used provided Training data. Then I collect more three laps on first track (two laps on the "wrong" way). I used Center, Left and Right Camera's images to training the model. To feed model with more data I used data augmentation strategy, using numpy flip function. Finally randomly shuffled the data set (code line 99) and put 20% of the data into a validation set (code line 42)

The overall strategy for deriving a model architecture was to use captured images and steering angles to train a Deep Neural Networks and use the trained model to drive a Car autonomously on simulator. 

My first step was to use a convolution neural network model similar to the [LeNet Architeture](http://yann.lecun.com/exdb/lenet/), but the result weren't goog enough. Then I change the model architeture creating a nes model based in [Nvidia's SelfDrive Car Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the mean squared error in training set decrease after each epoch, but after a certain level increase in  validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model inserting Dropout in Convolution and Fully Connected Layers (code line 118, 121, 123, 125, 128, 133, 135)

Then I trained model using Amazon's EC2 Intance, getting goog results:

![Alt text][training_aws]

![Alt text][chart]

The final step was to run the simulator to see how well the car was driving around track one. 

[Performance Video On Youtube](https://youtu.be/fX1CnW4eSz4) - Model driving Car.

Dowload a complete  [run](./run1.mp4) video. (car's point of view)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


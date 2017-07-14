# In[ ]:

# Importing libraries
import csv
import numpy as np
import os
import sklearn
from PIL import Image
from sklearn.model_selection import train_test_split
# import Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Importing csv with features and labels
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#define length
length=len(lines)

# delete collumns names        
lines=lines[1:length+1]

# creating validation data
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# defining a generator
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            
            images = []
            measurements = []
            correction = 0.2
	    # Using left and right images
            for line in batch_samples:

		# get file names and define paths
                source_path_center = line[0].split('/')[-1]
                source_path_left = line[1].split('/')[-1]
                source_path_right = line[2].split('/')[-1]

                path_center = 'data/IMG/' + source_path_center
                path_left = 'data/IMG/' + source_path_left
                path_right = 'data/IMG/' + source_path_right

		# Open Images and add to Images list
                image_center = Image.open(path_center)
                image_left = Image.open(path_left)
                image_right = Image.open(path_right)

                images.append(np.asarray(image_center))
                images.append(np.asarray(image_left))
                images.append(np.asarray(image_right))
		# Close images
                image_center.close()
                image_left.close()
                image_right.close()

		
		# Create lables with adjustments for left and right images
                measurement = float(line[3])
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)
            
	    # data Augmentation    
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
		# horizontal flip images
                augmented_images.append(np.fliplr(image))
                augmented_measurements.append(measurement*-1.0)
            
            # features and labels          
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Nvidia Architecture
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Cropping images (only see the road level)
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Convolutional, Pooling and Dropout layers
model.add(Convolution2D(24,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

# Define Loss and Optimizer
model.compile(loss='mse', optimizer='adam')

# Training Model
model.fit_generator(train_generator,
                    verbose=1,
                    samples_per_epoch= len(6*train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(6*validation_samples),
                    nb_epoch=10)

# save model
model.save('model.h5')


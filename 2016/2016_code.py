#!/usr/bin/env python3

"""
Begnin/malignant classification w/ ISIC 2016 challenge dataset
--------------------------------------------------------------
Link to challenge and dataset:
- 

Reference notebook: 
- https://www.kaggle.com/code/jagdmir/all-you-need-to-know-about-cnns
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import os

DATA_PATH = "2016/ISBI2016_ISIC_Part3_Training_Data/"


# LOAD CLASS DATA: from csv file, add img path
train = pd.read_csv("2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv", names=["image_name", "type"])
train["path"] = DATA_PATH + train.image_name + ".jpg"
print(f"Dataset shape: {train.shape}")
print(train.head)


# SPLIT DATA: train/test split
# .. (technically not needed here as ISIC provides a separate dataset for testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train, train.type, test_size=0.2) #random_state


# IMAGE DATA TO ARRAY
from keras.preprocessing.image import load_img, img_to_array

train_imgs = [img_to_array(load_img(img)) for img in x_train.path]
train_imgs = np.array(train_imgs)
train_labels = y_train
print(f"Train dataset shape: {train_imgs.shape}") ##

test_imgs = [img_to_array(load_img(img)) for img in x_test.path]
test_imgs = np.array(test_imgs)
test_labels = y_test
print(f"Test dataset shape: {test_imgs.shape}") ##


# SCALE IMAGES
train_imgs = train_imgs.astype("float32")
train_imgs /= 255
print(f"Scaled train dataset: {train_imgs.shape}") ##

test_img = test_imgs.astype("float32")
test_imgs /= 255
print(f"Scaled test dataset: {test_imgs.shape}") ##


"""
batch_size = total num images passed to model per iteration
weights = weights of units in layers are updaed after each iteration
epoch = when the whole dataset has passed through the network once
NB total num iterations per epoch = num training samples / batch_size
"""

# setup basic config
batch_size = None
num_classes = 2
epochs = None
input_shape = train_imgs.shape

"""
Basic CNN model here
---------------------
three convolutional layers
maxpooling for auto-extraction of features from images
downsampling the output convolution feature maps

Conv2D (see keras documentation)
-------------------------
keras.layers.Conv2D(
    filters,
        - determine the number of kernels to convolve with the input volume
        - the deeper the layer in the network, the more filters
        - depend on dataset complexity & depth of neural network
    kernel_size,
        - 2-tuple specifying width & height of 2D convolution window
        - typical values (1, 1) -odd-> (7, 7)
        - depends on input image size, network type & architecture complexity
    strides=(1, 1), 
        - 2-tuple of integers that specify the "step" of the convolution 
          along x & y axis of input volume
        - defaulted (1, 1), >1 = reduce output volume
    padding="valid", 
        - "valid" = no padding -> spatial dimensions naturally reduce in convolution
        - "same" = padding -> preserves spatial dimensions
    data_format=None, 
    dilation_rate=(1, 1), 
    groups=1, 
    activation=None, 
    use_bias=True, 
    kernel_initializer="glorot_uniform", 
    bias_initializer="zeros", 
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None, 
    kernel_constraint=None, 
    bias_constraint=None, 
    **kwargs
)

MaxPooling2d (see keras documentation)
------------
keras.layers.MaxPooling2D(
    pool_size=(2, 2), 
        - int or 2-tuple
        - window size to take the maximum from
    strides=None, 
        - int or 2-tuple or None
        - how far the pooling window moves for each pooling step
        - None = pool_size
    padding='valid',
        - "valid" = no padding
        - "same" = padding evenly so that output has same dimension as input
    data_format=None
        - ordering of dimensions in the inputs, default "channels_last"
        - "channels_last" = inputs with shape (batch, h, w, channels)
        - "channels_first" = inputs with shape (batch, channels, h, w)
)
"""

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers # purpose???

model = Sequential()

# ADD CONV LAYER
# .. create feature maps for each filter used,
# .. takes fm through activation function (relu)
model.add(
    Conv2D(16, # why 16?
           kernel_size=(3,3), # why (3,3)
           activation="relu", 
           input_shape=input_shape) # defined previously
)

# ADD MAXPOOLING LAYER
# .. select largest values on the feature maps
# .. then use as input for subsequent layers
model.add(MaxPooling2D(pool_size=(2,2))) # why (2,2)?

# ADD ANOTHER SET OF CONV & MAXPOOL
model.add(Conv2D(64, kernel_size=(3,3), activation="relu")) # need to edit + why 64?
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), activation="relu")) # need to edit + why 128?
model.add(MaxPooling2D(pool_size=(2,2)))

# FLATTEN -> function that converts the pooled feature map to a 1d array 
# .. to pass through fully-connected layer (final classification model)
model.add(Flatten())

# DENSE LAYER -> what is that??
model.add(Dense(512, activation="relu")) # why 512
model.add(Dense(1, activation="sigmoid")) # why 1 why sigmoid

# COMPILE MODEL
model.compile(loss="binary_crossentropy",
              optimizers=optimizers.RMSprop(),
              metrics=["accuracy"])

model.summary()


# to edit & understand, code copied from kaggle notebook
history = model.fit(x=train_imgs_scaled, y=train_labels,
                    validation_data=(validation_imgs_scaled, val_labels),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# .. model performance
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,31))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
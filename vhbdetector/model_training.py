# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:45:51 2022

@author: Muhammad Kaleemullah
"""
from data_loader import DataLoader
from models.cnn_rnn import conv_lstm

import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam


class TrainManager():
    def __init__(self):
        pass




dl = DataLoader()


X, Y = dl.load_data("datasets/X_normalized.pickle", "datasets/Y.pickle")
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
X.shape
# Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, shuffle = True, random_state = 5)


#del X
#del Y

# Model parameters
#CLASSES_LIST = list(range(Y_train.shape[1]))
#num_classes = len(CLASSES_LIST)
num_classes = Y_train.shape[1]
SEQUENCE_LENGTH = X_train.shape[1]
IMAGE_HEIGHT = X_train.shape[2]
IMAGE_WIDTH = X_train.shape[3]
channels = X.shape[-1]

# Define Hyperparameters
BATCH_SIZE = 24
EPOCHS = 250
learning_rate = 0.0001
callbacks_params = [
             callbacks.EarlyStopping(monitor='val_loss', mode = "min", patience=150),
             callbacks.ReduceLROnPlateau(patience = 15, verbose=1)]
             #callbacks.ModelCheckpoint('../input/honeybees/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)]


# Loding model
model = conv_lstm(seq_len = SEQUENCE_LENGTH, img_height = img_height, img_width = IMAGE_WIDTH, channels = channel)
print(model)

# Plot the structure of the contructed model and save it to corresponding file.
plot_model(model, to_file = 'models/cnn_rnn/conv_lstm/convlstm_model_structure_plot.png', show_shapes = True, show_layer_names = True)

# Compile the model and specify loss function, optimizer and metrics values to the model
convlstm_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ["accuracy"])



# Start training the model.
history = model.fit(x = X_train, y = Y_train, epochs = EPOCHS, batch_size = BATCH_SIZE,
                             shuffle = True, validation_data = (X_test, Y_test),
                             callbacks = callbacks_params)

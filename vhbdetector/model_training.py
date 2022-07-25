# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:45:51 2022

@author: Muhammad Kaleemullah
"""
from vhbdetector.data_loader import DataLoader
#from data_loader import DataLoader

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

data_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\wdd_ground_truth"
labels_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\ground_truth_wdd_angles.pickle"

#data, labels = dl.load_data_from_scratch(30, 25, 25, data_file, labels_file)

data, labels = dl.load_data("datasets/X_short.pickle", "datasets/Y_short.pickle")

from vhbdetector.pre_processing import Normalization, Preprocessing

norm = Normalization()

data = norm.pixels_normalization(data)

pp = Preprocessing()

data = pp.change_data_dimensions(data, 25, 15, 15)

#from vhbdetector.models.cnn_rnn.conv_lstm import CONVLSTM
#from models.cnn_rnn.conv_lstm_basic import BASIC_CONVLSTM
from models.cnn_rnn.convlstm_regularization_norms import CONVLSTM_REGR

#model = CONVLSTM(convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2, convolutions_drop_rate = 0.1, recurrent_drop_rate = 0.1)

#model = BASIC_CONVLSTM(convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2)

model = CONVLSTM_REGR(convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2, convolutions_drop_rate = 0.1, recurrent_drop_rate = 0.1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.3, stratify = labels, shuffle = True, random_state = 5)

model.create_model(X_train, Y_train, X_test, Y_test)

model.set_hyperparams(0.01, 8, 1)

history, trained_model = model.train_model()

accuracy = model.get_accuracy_score(X_test, Y_test)

video_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\sample_video.mp4"
video_pred = model.predict_video(video_file)

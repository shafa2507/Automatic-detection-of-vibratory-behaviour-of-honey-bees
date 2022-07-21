# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:48:06 2022

@author: Muhammad Kaleemullah
"""
from vhbdetector.data_loader import DataLoader
from vhbdetector.pre_processing import Preprocessing
from vhbdetector.models.base_model import BaseModel

import os
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam


class CONVLSTM(BaseModel):
    def __init__(self, convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2, convolutions_drop_rate = 0.1, recurrent_drop_rate = 0.1):
        # Layer parameters
        self.conv_actvn = convolutions_activation
        self.recurr_actvn = recurrent_activation
        self.td_drop_rate = time_distributed_drop_rate
        self.conv_drop_rate = convolutions_drop_rate
        self.recurr_drop_rate = recurrent_drop_rate
        
        #Validate Layer Parameters
        self.validate_params()
        
        # Data Arrays Declaration
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_val = np.array([])
        self.Y_val = np.array([])
        
        #Hyperparameters
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None
        self.early_stopping_epoch = None
        self.min_lr_decay = None
        self.lr_decay_mode = None
        self.checkpoints_save_path = None
        self.is_save_chkpt = False
        
        
        # Mode Information
        self.model = None
        self.is_model_built = False
        self.is_model_trained = False
        self.trained_model = None
        
        
        self.is_set_hyperparams = False
        
    def validate_params(self):
        activations = ["sigmoid", "tanh", "relu", "hard_sigmoid", "elu", "exponential", "gelu", "linear"]
        self.model = True
        if self.conv_actvn not in activations or self.recurr_actvn not in activations:
            raise Exception(f"Failed to load the model! The input activation function(s)  are invalid. Only the following activation functions are allowed to use in this model: {activations}.")
        elif self.td_drop_rate < 0 or self.td_drop_rate > 1 or self.conv_drop_rate < 0 or self.conv_drop_rate > 1 or self.recurr_drop_rate < 0 or self.recurr_drop_rate > 1:
            raise Exception("Dropout (1 - keep_probability) must be between 0 and 1!")
        return 0
        
    def create_model(self, X_train, Y_train, X_val, Y_val):
        '''
        This model has flexibility in taking different number of parameters.
        
        Parameters:
            
            #seq_len = 179, by default.    It should not be changed unless the videos data have sequence length more or less than 179. This parameter is used to provide flexibility for users if their videos have custom frame length.
            #img_height = 60, by default.    It should also be not changed unless there is a change in image size in data. This parameter is used to provide flexibility for users if their videos have custom image height.
            #img_width = 60, by default.    It should also be not changed unless there is a change in image size in data. This parameter is used to provide flexibility for users if their videos have custom image width.
            #channels = 1, by dedault,    used accordingly with the same shape of data. '1' means grayscale, if image has 3 dimenstions or channels (rgb), then channels = 3.
            #num_classes = 5.              According to the customized number of classes. It speficies how many classes model will train on.
            convolution_activation = "tanh", by default,      it is activation function of convolution layer.
            recurrent_activation = "hard_sigmoid", by default,      it is activation function of recurrent layer.
            time_distributed_drop_rate = 0.2, by dedault,           it controls how many (1 - keep_probabilities) features (units) in the network to drop.
            convolution_droo_rate = 0.1, by default,            it controls how many (1 - keep_probabilities) spacial features (units) in convolutions to drop.
            recurrent_drop_rate = 0.1, by default,             it controls how many (1 - keep_probabilities) temporal features (units) in recurrent network to drop.
            '''
            
        #Retreive the parameters for the layers
        convolutions_activation = self.conv_actvn
        recurrent_activation = self.recurr_actvn
        time_distributed_drop_rate = self.td_drop_rate
        convolutions_drop_rate = self.conv_drop_rate
        recurrent_drop_rate = self.recurr_drop_rate
        
        
        seq_len = X_train.shape[1]
        img_height = X_train.shape[2]
        img_width = X_train.shape[3]
        
        if len(X_train.shape) > 4 and len(X_val.shape) > 4:
            channels = X_train.shape[-1]
        else:
            X_train = X_train.reshape(X_train.shape[0], seq_len, img_height, img_width, 1)
            X_val = X_val.reshape(X_val.shape[0], seq_len, img_height, img_width, 1)
            channels = X_train.shape[-1]
        
        if len(Y_train.shape) == 1:
            Y_train = to_categorical(Y_train)
        elif len(Y_train.shape) == 2:
            if Y_train.shape[1] == 1:
                Y_train = to_categorical(Y_train.reshape(seq_len, -1))
            
        if len(Y_val.shape) == 1:
            Y_val = to_categorical(Y_val)
        elif len(Y_val.shape) == 2:
            if Y_val.shape[1] == 1:
                Y_val = to_categorical(Y_val.reshape(seq_len, -1))
            
        num_classes = len(list(range(Y_train.shape[1])))
        
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
            
            
            
        # We will use a Sequential model for model construction
        model = Sequential()
            
        # Define the Model Architecture.
        ########################################################################################################################
        # CONVLSTM Layer 1
        model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = convolutions_activation,
                         recurrent_activation = recurrent_activation, data_format = "channels_last",
                         dropout = convolutions_drop_rate, recurrent_dropout= recurrent_drop_rate, return_sequences=True,
                         input_shape = (seq_len, img_height, img_width, channels)))
            
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
            
            
        # CONVLSTM Layer 2
        model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = convolutions_activation,
                         recurrent_activation = recurrent_activation, data_format = "channels_last",
                         dropout = convolutions_drop_rate, recurrent_dropout = recurrent_drop_rate, return_sequences=True))
            
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
            
            
        # CONVLSTM Layer 3
        model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = convolutions_activation,
                         recurrent_activation = recurrent_activation, data_format = "channels_last",
                         dropout = convolutions_drop_rate, recurrent_dropout= recurrent_drop_rate, return_sequences=True))
        
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
            
            
        # CONVLSTM Layer 4
        if img_height >= 60 or img_width >= 60:
            model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), activation = convolutions_activation,
                         recurrent_activation = recurrent_activation, data_format = "channels_last",
                         recurrent_dropout= recurrent_drop_rate, return_sequences=True))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
            model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
        model.add(Flatten()) 
            
        
        model.add(Dense(num_classes, activation = "softmax"))
        ########################################################################################################################
        # Display the models summary
        model.summary()
        
        self.is_model_built = True
        self.model = model
        
        # Return the constructed convlstm model.
            
        return model
    
    def set_hyperparams(self, learning_rate = 0.001,  batch_size = 32, epochs = 100, early_stopping_epoch = 0, lr_decay_rate = 0, min_lr_weight_decay = 0, checkpoints_save_path = None):
        """
        Optimizers : Adam, by default, because Adam works best and outperforms other gradient descent algorithms.
        """
        
        
        
        save_chkpt, chkpt_path_valid = self.validate_hyperparams(learning_rate,  batch_size, epochs, early_stopping_epoch, lr_decay_rate, min_lr_weight_decay, checkpoints_save_path)
        
        # Update hyperparameters to class istances
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_epoch = early_stopping_epoch
        self.lr_decay_rate = lr_decay_rate
        self.min_lr_decay = min_lr_weight_decay
        if save_chkpt:
            self.is_save_chkpt = True
            if chkpt_path_valid:
                self.checkpoints_save_path = checkpoints_save_path
        
        self.is_set_hyperparams = True
            
            
    def validate_hyperparams(self, learning_rate = 0.001,  batch_size = 32, epochs = 100, early_stopping_epoch = 0, lr_decay_rate = 0, min_lr_weight_decay = 0, checkpoints_save_path = None):
        
        save_chkpt = False
        chkpt_path_valid = False
        
        
        if learning_rate > 0.99 or learning_rate < 0.000000001:
            raise Exception("Learnig rate should be between 0.99 and 0.000000001!")
        
        elif batch_size < 2 or batch_size > 1000:
            raise Exception (f"{batch_size} is useless for this problem. Allowed interval is (2, 1000), but it's always preffered to use between 32 to 512 for smaller datasets like this.")
        elif epochs < 1:
             raise Exception("Epochs cannot be less than 1.")
        elif early_stopping_epoch < 0 or early_stopping_epoch > epochs:
            raise Exceptio(f"{early_stopping_epoch} cannot be used. It cannot be negative nor exceeds the total number of epochs.")
        elif lr_decay_rate > learning_rate or lr_decay_rate < 0:
            raise Exception(f"Got decay rate of {lr_decay_rate}! It must be less learning rate and not be less than 0.")
        elif min_lr_weight_decay > learning_rate or min_lr_weight_decay < 0:
            raise Exception(f"Got min_lr_weight_decay of {min_lr_weight_decay}! It must be less than learning rate and greater than 0.")
        elif checkpoints_save_path:
            # It means user wants to save model checkpoints
            save_chkpt = True
            if checkpoints_save_path == True:
                pass
            elif os.path.exists(checkpoints_save_path):
                chkpt_path_valid = True
        return save_chkpt, chkpt_path_valid
        
    
    
    def train_model(self):
        
        
        if not self.is_model_built:
            raise Exception("Please create model by object.create_model() function of this class first before training the model!")
        
        if not self.is_set_hyperparams:
            self.set_hyperparams()
        
        
        print(f"X_train.shape = {self.X_train.shape}")
        print(f"X_val.shape = {self.X_val.shape}")
        print(f"Y_train.shape = {self.Y_train.shape}")
        print(f"Y_val.shape = {self.Y_val.shape}")      
        
        
        
        print(f"learning_rate: {self.learning_rate}")
        print(f"batch_size: {self.batch_size}")
        print(f"epochs: {self.epochs}")
        print(f"early_stopping_epoch: {self.early_stopping_epoch}")
        print(f"lr_decay_rate {self.lr_decay_mode}")
        print(f"min_lr_decay: {self.min_lr_decay}")
        print(f"is_save_checkpoints: {self.is_save_chkpt}")
        print(f"checkpoints_save_path: {self.checkpoints_save_path}")        
        print(f"is_set_hyperparameters {self.is_set_hyperparams}")
        
        
        
        callbacks_params = []
        if self.early_stopping_epoch > 0:
            callback_params.append(EarlyStopping(monitor='val_loss', mode = "min", patience = self.early_stopping_epoch))
        if self.lr_decay_rate > 0:
            callbacks_params.append(callbacks.ReduceLROnPlateau(patience = int(self.early_stopping_epoch / 4), verbose=1, min_lr = self.min_lr_decay))
        
        if self.checkpoints_save_path:
            callbacks_params.append(ModelCheckpoint(os.path.join(self.checkpoints_save_path), "\weights.{epoch:02d}-{val_loss:.2f}.hdf5"), verbose=1)
        elif self.is_save_chkpt:
            callbacks_params.append(ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1))
        
        print(callbacks_params)
        
        
        # Comiple Model
        self.model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = self.learning_rate), metrics = ["accuracy"])
        
        
        
        if np.size(callbacks_params):
            # Start training the model with having callbacks
            history = self.model.fit(x = self.X_train, y = self.Y_train, epochs = self.epochs, batch_size = self.batch_size,
                             shuffle = True, validation_data = (self.X_val , self.Y_val),
                             callbacks = callbacks_params)
        else:
            # Start training the model without having callbacks
            history = self.model.fit(x = self.X_train, y = self.Y_train, epochs = self.epochs, batch_size = self.batch_size,
                             shuffle = True, validation_data = (self.X_val , self.Y_val))
        
        
        self.history = history
        self.trained_model = self.model
        
        self.is_model_trained = True
        super().__init__(self.X_train, self.Y_train, self.X_val, self.Y_val, self.history, self.trained_model, True)
        
        return history, self.model
        


if __name__ == "__main__":
    #dl = DataLoader()
    #data, labels = dl.load_data("datasets/X_normalized_temp.pickle", "datasets/Y_temp.pickle")
    
    model = CONVLSTM(convolutions_drop_rate=0.2, recurrent_drop_rate=0.2, time_distributed_drop_rate=0.3)
    
    
    dl = DataLoader()
    data_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\X_short.pickle"
    labels_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\Y_short.pickle"
    data, labels = dl.load_data(data_file, labels_file)
    
    video_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\sample_video.mp4"
    
    from vhbdetector.pre_processing import Preprocessing
    pp = Preprocessing()
    print(data.shape)
    data = pp.change_data_dimensions(data, 60, 30, 30)    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.3, stratify = labels, shuffle = True)
    
    copy = model.create_model(X_train, Y_train, X_test, Y_test)
    
    
    model.set_hyperparams(0.0001, 16, 1, 0, 0, 0, True)
    history, trained_model = model.train_model()
    accuracy = model.get_accuracy_score(X_test, Y_test)
    model.save_model("save_model")
    model.save_train_val_curves("accuracy", "loss")
    video_output = model.predict_video(video_file)
    #print(f"data shape = {data.shape}")
    #print(f"labels shape = {labels.shape}")
    
    #sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    #print("absolute path: ", os.path.dirname(os.path.abspath(__file__)))
    #simp_path = __file__
    #abs_path = os.path.abspath(simp_path)
    #print(abs_path)
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, Y_train, Y_test = train_test_split(data, labels, stratify = labels, test_size = 0.3, random_state = 5)
    #new_model = model.create_model(Y_train, X_test, Y_train, Y_test)
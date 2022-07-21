
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


class CNN_TD(BaseModel):
    """
    Filters will be multiplied by 2 per next layer, so don't use more than 128
    """
    def __init__(self, filters = 8, convolutions_activation = "tanh", drop_rate = 0.1):
        # Layer parameters
        self.conv_actvn = convolutions_activation
        self.drop_rate = drop_rate
        self.filters = filters
        
        #Validate Layer Parameters
        self.validate_params()
        
        # Data Arrays Declaration
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_val = np.array([])
        self.Y_val = np.array([])
        self.num_classes = None
        
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
        if self.conv_actvn not in activations:
            raise Exception(f"Failed to load the model! The input activation function(s)  are invalid. Only the following activation functions are allowed to use in this model: {activations}.")
        elif  self.drop_rate < 0 or self.drop_rate > 1:
            raise Exception("Dropout (1 - keep_probability) must be between 0 and 1!")
        elif self.filters < 1:
            raise Exception("Filters cannot be zero! Please define class object again!")
        elif self.filters > 128:
            raise Exception("Filters will be multiplied by 2 per next layer, so don't use more than 128!")
        return 0
    
    def build_convnet(self, shape):
        momentum = .9
        model = Sequential()
        model.add(Conv2D(self.filters, (3,3), input_shape=shape, padding='same', activation=self.conv_actvn))
        model.add(Conv2D(self.filters * 2, (3,3), padding='same', activation=self.conv_actvn))
        model.add(BatchNormalization(momentum=momentum))
        
        model.add(MaxPool2D())
        
        model.add(Conv2D(self.filters *2, (3,3), padding='same', activation=self.conv_actvn))
        model.add(Conv2D(self.filters * 2, (3,3), padding='same', activation=self.conv_actvn))
        model.add(BatchNormalization(momentum=momentum))
        
        model.add(MaxPool2D())
        
        if shape[0] > 10 or shape[1] > 10:
            model.add(Conv2D(self.filters * 2, (3,3), padding='same', activation = self.conv_actvn))
            model.add(Conv2D(self.filters * 2, (3,3), padding='same', activation = self.conv_actvn))
            model.add(BatchNormalization(momentum=momentum))
        
        model.add(MaxPool2D())
        
        model.add(Conv2D(self.filters * 4, (3,3), padding='same', activation=self.conv_actvn))
        model.add(Conv2D(self.filters * 4, (3,3), padding='same', activation=self.conv_actvn))
        model.add(BatchNormalization(momentum=momentum))
        
        # flatten...
        model.add(GlobalMaxPool2D())
        return model

    def action_model(self, shape):
        if not self.num_classes:
            raise Exception("Please create model first and then train model using only class object!")
        # Create our convnet with (112, 112, 3) input shape
        convnet = self.build_convnet(shape[1:])
        
        # then create our final model
        model = Sequential()
        # add the convnet with (5, 60, 60, 1) shape
        model.add(TimeDistributed(convnet, input_shape=shape))
        # and finally, we make a decision network
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model
        
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
        
        self.num_classes = num_classes
        
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        
        
        
            
            
            
        model = self.action_model(shape = (seq_len, img_height, img_width, channels))
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
    
    model = CNN_TD(1)
    
    
    dl = DataLoader()
    data_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\X_short.pickle"
    labels_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\Y_short.pickle"
    data, labels = dl.load_data(data_file, labels_file)
    
    
    from vhbdetector.pre_processing import Preprocessing
    pp = Preprocessing()
    print(data.shape)
    data = pp.change_data_dimensions(data, 4, 25, 25) 
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.3, stratify = labels, shuffle = True)
    
    copy = model.create_model(X_train, Y_train, X_test, Y_test)
    
    history, trained_model = model.train_model()








# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:48:06 2022

@author: Muhammad Kaleemullah
"""
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1, L2
def convlstm_regnorm_model(seq_len = 179, img_height = 60, img_width = 60, channels = 1, num_classes = 5, convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2, convolutions_drop_rate = 0.1, recurrent_drop_rate = 0.1, regularization = "l2", regularization_penalty = 0.001, num_convlstm_layers = 4, use_batch_norm = True):
    
    '''
    This function will construct the required convlstm model. It supports l1 and l2 regularization and provides more flexibility.
    than other cnn-rnn models. It also contains the advantage of Batch Normalization.
    Parameters:
        seq_len = 179, by default. It should not be changed unless the videos data have sequence length more or less than 179. This parameter is used to provide flexibility for users if their videos have custom frame length.
        img_height = 60, by default. It should also be not changed unless there is a change in image size in data. This parameter is used to provide flexibility for users if their videos have custom image height.
        img_width = 60, by default. It should also be not changed unless there is a change in image size in data. This parameter is used to provide flexibility for users if their videos have custom image width.
        channels = 1, by dedault, used accordingly with the same shape of data. '1' means grayscale, if image has 3 dimenstions or channels (rgb), then channels = 3.
        num_classes = 5. According to the customized number of classes. It speficies how many classes model will train on.
        convolution_activation = "tanh", by default, it is activation function of convolution layer.
        recurrent_activation = "hard_sigmoid", by default, it is activation function of recurrent layer.
        time_distributed_drop_rate = 0.2, by dedault, it controls how many (1 - keep_probabilities) features (units) in the network to drop.
        convolution_droo_rate = 0.1, by default, it controls how many (1 - keep_probabilities) spacial features (units) in convolutions to drop.
        recurrent_drop_rate = 0.1, by default, it controls how many (1 - keep_probabilities) temporal features (units) in recurrent network to drop.
        regularization = l2, by default. It supports two popular types of regularizations, l1 and l2.
        regularization_penalty = 0.001, by default. It penalize the weights with the value given in regularization penalty.
        num_convlstm_layers = 4, by default, it specifies how many layers of convlstm2d to use.
        use_batch_norm = True, by default, it uses Batch Normalization by deafult but it can be set False to ignore. 
        
    Returns:
        model: It is the required constructed convlstm model.
    '''
    
    
    activations = ["sigmoid", "tanh", "relu", "hard_sigmoid", "elu", "exponential", "gelu", "linear"]
    regularization_types = ["l1", "l2"]
    if convolutions_activation not in activations or recurrent_activation not in activations or time_distributed_drop_rate < 0 or time_distributed_drop_rate > 1 or convolutions_drop_rate < 0 or convolutions_drop_rate > 1 or recurrent_drop_rate < 0 or recurrent_drop_rate > 1:
        raise Exception("Failed to load the model! The activation functions or dropout values are invalid.")
    elif num_convlstm_layers < 1:
        raise Exception("Failed to load the model! Please input valid number of layers.")
    elif regularization not in regularization_types:
        raise Exception("Failed to load the model! Regularization is invalid or unknown. Please enter it again.")
    
    
    if regularization == "l1":
        reg = L1(regularization_penalty)
    else:
        reg = L2(regularization_penalty)
        
        
    # We will use a Sequential model for model construction
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################
    for i in range(0, num_convlstm_layers):
        model.add(ConvLSTM2D(filters = 8 + i**2, kernel_size = (3, 3), activation = convolutions_activation,
                         recurrent_activation = recurrent_activation, data_format = "channels_last",
                         dropout = convolutions_drop_rate, recurrent_dropout= recurrent_drop_rate, return_sequences=True,
                         kernel_regularizer = reg, recurrent_regularizer = reg,
                         input_shape = (seq_len, img_height, img_width, channels)))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
        
    model.add(Flatten()) 
    model.add(Dense(num_classes, activation = "softmax"))
    
    ########################################################################################################################
    # Display the models summary.
    model.summary()
    
    # Return the constructed convlstm model.
    return model

model = convlstm_reg_model(num_convlstm_layers = 1)

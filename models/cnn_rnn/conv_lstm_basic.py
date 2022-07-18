from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

def convlstm_model_no_droput(seq_len = 179, img_height = 60, img_width = 60,  channels = 1, num_classes = 5, convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.1):
    '''
    This function will construct the required convlstm model. As it name suggests it's the basic structure consists of 4 convlsts2d layers that captures spatio-temporal features better than the other models.
    Parameters:
        seq_len = 179, by default. It should not be changed unless the videos data have sequence length more or less than 179. This parameter is used to provide flexibility for users if their videos have custom frame length.
        img_height = 60, by default. It should also be not changed unless there is a change in image size in data. This parameter is used to provide flexibility for users if their videos have custom image height.
        img_width = 60, by default. It should also be not changed unless there is a change in image size in data. This parameter is used to provide flexibility for users if their videos have custom image width.
        channels = 1, by dedault, used accordingly with the same shape of data. '1' means grayscale, if image has 3 dimenstions or channels (rgb), then channels = 3.
        num_classes = 5. According to the customized number of classes. It speficies how many classes model will train on.
        convolution_activation = "tanh", by default, it is activation function of convolution layer.
        recurrent_activation = "hard_sigmoid", by default, it is activation function of recurrent layer.
    Returns:
        model: It is the required constructed convlstm model. It returns the customized model with different parameters.
    '''
    
    activations = ["sigmoid", "tanh", "relu", "hard_sigmoid", "elu", "exponential", "gelu", "linear"]
    if convolutions_activation not in activations or recurrent_activation not in activations or time_distributed_drop_rate < 0 or time_distributed_drop_rate > 1:
        raise Exception("Failed to load the model! The activation functions or dropout values are invalid.")
    
    # We will use a Sequential model for model construction
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################
    
    model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = convolutions_activation, data_format = "channels_last",
                         recurrent_dropout=0, return_sequences=True, recurrent_activation = recurrent_activation,
                         input_shape = (seq_len, img_height, img_width, channels)))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
    
    model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = convolutions_activation, data_format = "channels_last",
                         recurrent_dropout=0, return_sequences=True, recurrent_activation = recurrent_activation))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
    
    model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = convolutions_activation,
                         recurrent_activation = recurrent_activation, data_format = "channels_last",
                         recurrent_dropout=0, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))

    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), activation = convolutions_activation,
                         data_format = "channels_last", recurrent_activation = recurrent_activation,
                         recurrent_dropout=0, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(time_distributed_drop_rate)))
    
    model.add(Flatten())

    ##model.add(Dense(65536, activation = "relu"))

    model.add(Dense(1024, activation = "relu"))

    model.add(Dense(512, activation = "relu"))
    
    model.add(Dense(num_classes, activation = "softmax"))
    
    ########################################################################################################################
     
    # Display the models summary.
    model.summary()
    
    # Return the constructed convlstm model.
    return model


model = convlstm_model_no_droput(convolutions_activation="relu")
#model = convlstm_model_no_droput(179, img_height, img_width, convolutions_activation, recurrent_activation, num_classes, conv_droput, recurrent_dropout)
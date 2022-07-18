from tensorflow.keras.layers import *
from tensorflow.models import Sequential

def convolutions2d_model(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES):
    '''
    This function will construct the required convlstm model.
    Returns:
        model: It is the required constructed convlstm model.
    '''
    
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same', activation = 'relu' ,
                     input_shape = (IMAGE_HEIGHT,IMAGE_WIDTH, 1)))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same', activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    
    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same', activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    
    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same', activation = 'relu'))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    
    model.add(Lambda(ReshapeLayer))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    
    
    return model


def ReshapeLayer(x):
    
    shape = x.shape
    
    # 1 possibility: H,W*channel
    reshape = Reshape((shape[1],shape[2]*shape[3]))(x)
    
    # 2 possibility: W,H*channel
    # transpose = Permute((2,1,3))(x)
    # reshape = Reshape((shape[1],shape[2]*shape[3]))(transpose)
    
    return reshape
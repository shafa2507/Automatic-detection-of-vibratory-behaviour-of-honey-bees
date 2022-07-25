from vhbdetector.pre_processing import Preprocessing

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


class BaseModel():
    """
    Here, we are using BaseModel class to evaluate model and reduce redundancy and improve reproducibility
    """
    def __init__(self, X_train, Y_train, X_val, Y_val, history, trained_model, is_trained):
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.trained_model = trained_model
        self.is_model_trained = is_trained
        self.history = history
    
    def save_model(self, save_file = None):
        if not self.is_model_trained:
            raise Exception("Please tran the model first to save the model")
        try:
            if save_file:
                self.trained_model.save('saved_model')
                print("Model has been saved successfully!")
            else:
                self.trained_model.save('saved_model')
                print("Model has been saved successfully!")
        except:
            print("Error in file name")
            
            
            
    def load_model(self, load_file):
        if os.path.exists(load_file):
            try:
                self.trained_model = load_model(load_file)
            except:
                print("Error in loadinf model, please try again with valid file!")
            self.is_model_trained = True
        
        
    def get_accuracy_score(self, X_test, Y_test):
        
        from sklearn.metrics import accuracy_score
        
        if not self.is_model_trained:
            raise Exception("Please tran the model first to save the model")
            
        print(X_test.shape)
        print(Y_test.shape)
        
        try:
            if len(Y_test.shape) == 1:
                Y_test = to_categorical(Y_test)
            elif len(Y_test.shape) == 2:
                if Y_test.shape[1] == 1:
                    Y_test = to_categorical(Y_test.reshape(Y_test.shape[0], -1))
        except:
            print("Error in reshape the Y_test array, seems like it's corrput on not appropriate, dimensions different.")
            
        try:
            if len(X_test.shape)  == 5:
                if X_test.shape[4] == 1:
                    pass
            elif len(X_test.shape) == 4:
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)
        except:
            print("Error in reshape the Y_test array, seems like it's corrput on not appropriate, dimensions different.")
        
        Y_test = np.argmax(Y_test, axis = 1)
        Y_pred = self.predict_multiple_samples(X_test)
        
        print(f"This model has got the Accuracy Score: {accuracy_score(Y_test, Y_pred)}.")
        
        return accuracy_score(Y_test, Y_pred)
    
    def predict_multiple_samples(self, X_test):
        """
        Input samples in only parameter
        """
        
        if not self.is_model_trained:
            raise Exception("Please tran the model first to save the model")
            
        try:
            if len(X_test.shape)  == 5:
                if X_test.shape[4] == 1:
                    pass
            elif len(X_test.shape) == 4:
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)
        except:
                print("Error in reshape the X_test array, seems like it's corrput on not appropriate, dimensions different.")
            
        Y_pred_probabilities = self.trained_model.predict(X_test)
        Y_pred = np.argmax(Y_pred_probabilities, axis = 1)
        return Y_pred
    
    
    def save_train_val_curves(self, accuracy_path, loss_file_path):
        import matplotlib.pyplot as plt
        """
        It saves Accuracy Curve in a given accuracy path and Loss Curve in a given loss_path
        """
        if not self.is_model_trained:
            raise Exception("Please train the model first to save the model")
            
        try:
            # summarize history for accuracy
            plt.figure(figsize = (8, 6), dpi = 300)
            plt.tight_layout()
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
            plt.savefig(fname = accuracy_path, format = "png")
            plt.show()
        except:
            print("Please use the appropriate name for Accuracy File to save accuracy curve.")
        
        try:
            # summarize history for loss
            plt.figure(figsize = (8, 6), dpi = 300)
            plt.tight_layout()
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
            plt.savefig(fname = loss_file_path, format = "png")
            plt.show()
        except:
            print("Please use the appropriate name for Loss File to save loss curve.")
        print("Figures are saved successfully!")
        
        
    
    def predict_video(self, video_path):
        import cv2
        video = []
        if not self.is_model_trained:
            raise Exception("Please train the model first to save the model")
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            while success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.X_train.shape[2], self.X_train.shape[3]))
                video.append(frame)
                success, frame = cap.read()
        video = np.array(video)
        video = video.reshape(1, video.shape[0], video.shape[1], video.shape[2], 1)
        pp = Preprocessing()
        video = pp.change_data_dimensions(video, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3])
        
        
        video_pred_probability = self.trained_model.predict(video)
        video_pred = np.argmax(video_pred_probability, axis = 1)
        
        video_labels = ["activating", "other", "trembling", "ventilating", "waggle"]
        
        if video_labels[video_pred[0]] == "waggle":
            print(f"In this video, the Honeybees are {video_labels[video_pred[0]]} dancing!")
        else:
            print(f"In this video, the Honeybees are {video_labels[video_pred[0]]}!")
        
        return video_labels[video_pred[0]]
    
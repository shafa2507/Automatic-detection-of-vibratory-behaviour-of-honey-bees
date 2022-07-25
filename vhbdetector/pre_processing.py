# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 00:53:45 2022

@author: Muhammad Kaleemullah
"""

from vhbdetector.data_loader import DataLoader

import numpy as np
import pickle
class Normalization():
    def __init__(self):
        pass
        
    def pixels_normalization(self, videos_data, path_to_save = None):
        """
        Parameters:
            videos_data: It gets input of videos data organized in numpy array.
            path_to_same: It get the filename or fullpath(valid) of file where it should be saved.
            Returns: According to the filename or fullpath(valid) given in the parameter path_to_save, it can save the normalized data as well as it will return the data.
        """
        if path_to_save:
            if path_to_save == True or (len(path_to_save) < 6 or path_to_save[-7:] != ".pickle"):
                pickle.dump(videos_data, open("X_new.pickle", "wb"))
            else:
                try:
                    videos_data = videos_data / 255
                    pickle.dump(videos_data, open(path_to_save, "wb"))
                except:
                    print("Error in either file name or file path!")
        else:
            videos_data = videos_data / 255
        return videos_data






class Preprocessing():
    def __init__(self):
        pass
    def change_data_dimensions(self, data, to_seq_len = 100, to_img_height = 60, to_img_width = 60):
        """
        It cuts frame from sides (from start and end) and reduce picels corresponding to the size given.
        """
        
        #Only relevant if it get the channel dimension
        reshape = False
        
        import cv2
        
        try:
            if len(data.shape) > 4:
                # Only change if in case of 5 dimensions
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
                reshape = True
        except:
            Exception("Video data as input got more than required dimensions. Please use appropriate data!")
        
        if to_seq_len > data.shape[1] or to_img_height > data.shape[2] or to_img_width > data.shape[3]:
            raise Exception(f"Please input valid shape of dimensions! The data has already these {(data.shape[1], data.shape[2], data.shape[3])} dimensions. You probably are entering more size than it already has.")
        #print(data.shape)
        subt_frame = (data.shape[1] - to_seq_len) / 2
        
        
        #print(f"{round(subt_frame + 0.1)} : {data.shape[1] - int(subt_frame)}")
        #print(to_seq_len)
        #print(round(to_seq_len - subt_frame + 0.1))
        #print(f"{round(subt_frame + 0.1)} subt_frame_len")
        videos_data = np.array([])
        for video_idx in range(0, data.shape[0]):
            video = []
            for seq_idx in range(0, data.shape[1]):
                if seq_idx >= subt_frame and seq_idx < (data.shape[1] - int(subt_frame)):
                    #print(cv2.resize(data[video_idx][seq_idx], dsize = (img_height, img_width)))
                    video.append(cv2.resize(data[video_idx][seq_idx], dsize = (to_img_height, to_img_width)))
                    #video.append(data[video_idx][seq_idx].resize(to_img_height, to_img_width))
                
            
            #print(np.array(video).reshape(1, to_seq_len, 60, 60).shape)
            if videos_data.size == 0:
                #print(np.array(video).shape)
                videos_data = np.array(video).reshape(1, to_seq_len, to_img_height, to_img_width)
                continue
            videos_data = np.append(videos_data, np.array(video).reshape(1, to_seq_len, to_img_height, to_img_width), axis = 0)
            
            
        if reshape:
            videos_data = videos_data.reshape(videos_data.shape[0], videos_data.shape[1], videos_data.shape[2], videos_data.shape[3], 1)
        
        #Check if we got the required length
        if videos_data.shape[1] == to_seq_len and videos_data.shape[2] == to_img_height and videos_data.shape[3] == to_img_width:
            print("Dimensions of video data changed successfully!")
        return videos_data
                


if __name__ == "__main__":
    dl = DataLoader()
    data, target = dl.load_data("datasets/X_short.pickle", "datasets/Y_short.pickle")
    
    norm = Normalization()
    
    data = norm.pixels_normalization(data, "True.pickle")
    
    pp = Preprocessing()
    
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1)
    videos_data = pp.change_data_dimensions(data, 100, 30, 30)
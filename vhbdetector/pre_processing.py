# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 00:53:45 2022

@author: Muhammad Kaleemullah
"""

from vhbdetector.data_loader import DataLoader
import tensorflow as tf

import numpy as np
import pickle


class Augmentation():
    def __init(self):
        pass
    
    def balance_data(self, data, labels):
        try:
            assert data.shape[0] == labels.shape[0]
            
            if len(data.shape) > 3:
                frame_len = data.shape[1]
                img_height = data.shape[2]
                img_width = data.shape[3]
                pass
            else:
                raise Exception("The video data is either corrputed or not loaded completely, it does not contain complete information!")
        except:
            print("Data and labels are not equal in length! Please try again with appropriate shape!")
        
        unique_labels = set(labels)
        labels_count = {}
        for i in unique_labels:
            labels_count[i] = np.count_nonzero(labels == i)
        
        
        import operator
        labels_count_ordered = dict(sorted(labels_count.items(), key=operator.itemgetter(1),reverse=True))
        
        
        add_labels_count = {}
        
        count = 0
        for key in labels_count.keys():
            if count > 0:
                add_labels_count[key] = labels_count_ordered[0] - labels_count_ordered[key]
            count += 1
        
        augmented_frame = 0
        videos_data = np.array([])
        new_labels = np.array([])
        
        
        for class_ in range(len(add_labels_count.keys())):
            count = 0
            
            for idx in range(len(labels)):
                video = []
                is_video_augmented = False
                
                if (labels[idx] == (class_ + 1)) and (count < add_labels_count[class_ + 1]):
                    for frame in range(data[idx].shape[0]):
                        #print("------------\n\n\n Index = ", idx)
                        #print("Shape of data[idx]", data[idx].shape)
                        #print("Frame number = ", frame)
                        
                        #print("\n Data Shape = ", data[idx][frame].reshape(img_height, img_width, 1).shape)
                        augmented_frame = tf.image.rot90(data[idx][frame].reshape(img_height, img_width, 1), k = 2)
                        augmented_frame = augmented_frame.numpy()
                        #print("Augmented frame shape = ", augmented_frame.shape)
                        
                        #print("Augmented frame shape after reshaping it = ", augmented_frame.reshape(img_height, img_width).shape)
                        
                        video.append(augmented_frame)
                        #print("Video shape = ", np.array(video).shape)
                        is_video_augmented = True
                        
                    #print("Label's value = ", labels[idx])
                    count += 1
                if is_video_augmented:
                    new_labels = np.append(new_labels, labels[idx])
                    if videos_data.size == 0:
                        
                        videos_data = np.array(video).reshape(1, data.shape[1], img_height, img_width)
                        continue
                        
                    videos_data = np.append(videos_data, np.array(video).reshape(1, data.shape[1], img_height, img_width), axis  = 0)
                    #print("Videos shape = ", videos_data.shape)
                   
            #print("Total Instances added = ", count)
            #print("Total Instances required = ", add_labels_count[class_ + 1])
        
        print("data.shape = ", data.shape)
        print("videos_data.shape = ", videos_data.shape)
        data = np.append(data, videos_data, axis = 0)
        labels = np.append(labels, new_labels, axis = 0)
        
        print("-------------After update-------------------")
        print("data.shape = ", data.shape)
        print("labels.shape = ", labels.shape)
        
        return (data, labels)
        
            
        
        pass

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
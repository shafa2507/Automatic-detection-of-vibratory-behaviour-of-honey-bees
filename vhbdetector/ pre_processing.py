# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 00:53:45 2022

@author: Muhammad Kaleemullah
"""
import numpy as np
import pickle
class Normalization():
    def __init__(self):
        self.data = np.array([])
    def pixels_normalization(self, videos_data, path_to_save):
        videos_data = videos_data / 255
        pickle.dump(videos_data, open(path_to_save, "wb"))

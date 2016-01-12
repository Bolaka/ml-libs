# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:03:07 2015

@author: abzooba
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeatureMixer(object):
    
    def __init__(self, dataRover):
        
        self.directory = dataRover.directory
        self.train = dataRover.train
        self.test = dataRover.test
        self.categoricals = dataRover.categoricals
        self.numericals = dataRover.numericals
        self.targets = dataRover.targets
        self.id_col = dataRover.id_col
        
    def outlierCleaning(self):
        pass
    
    def categoryEncoder(self):
        pass
    
    def missingHandler(self):
        self.train = self.train.fillna(0)
        self.test = self.train.fillna(0)
    
    def preProcessing(self):
        print 'Generic feature mixer...'        
        
        # pre-process categoricals
        for cate in self.categoricals:
            column = self.train[cate]
            dtype = str(column.dtype)
            if dtype == 'object' and cate not in self.targets:
                le = LabelEncoder()
                le.fit(np.append(self.train[cate], self.test[cate]))
            
                self.train[cate] = le.transform(self.train[cate])
                self.test[cate] = le.transform(self.test[cate])
        
        self.features = self.categoricals + self.numericals
        
        for t in self.targets:
            self.features.remove(t)
            
        return True
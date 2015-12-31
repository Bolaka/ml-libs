# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:11:42 2015

@author: abzooba
"""
from __future__ import division
import imp
import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy import stats
from numpy.random import seed
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import LabelEncoder
#import xgboost as xgb

# suppress pandas warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

datarover = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/datarover.py')
preprocessing = imp.load_source('preprocessing', '/home/abzooba/python-workspace/libs/preprocessing.py')
predictors = imp.load_source('predictors', '/home/abzooba/python-workspace/libs/predictors.py')

class MachineLearningPipeline(object):
    
    def __init__(self, dataRover, featureMixer):
        
        # reproduce results
        seed(786)           
        
        self.dataRover = dataRover
        self.featureMixer = featureMixer
        
        # the data source
        self.directory = dataRover.directory
        self.train = dataRover.train
        self.test = dataRover.test

        self.memory_usage = dataRover.memory_usage
        
        print 'train data : ' + str(self.train.shape[0]) + ' rows, ' + str(self.train.shape[1]) + ' columns'                
        print 'test data : ' + str(self.test.shape[0]) + ' rows, ' + str(self.test.shape[1]) + ' columns'
        print 'total memory usage : ' + str(self.memory_usage/1024.0) + ' (KB)'
    
    def exploringData(self, setID):
#        self.dataRover = DataRover(self.directory, self.train, self.test)
        
        self.data_valid = self.dataRover.extractMetadata(setID)        
        
#        if self.data_valid != False:
#            self.dataRover.univariate()
#            self.dataRover.bivariateTarget()
        #    ml.bivariate('season', 'weather')
        #    ml.bivariate('season', 'temp')
        #    ml.bivariate('temp', 'atemp')        
        
        self.id_col = self.dataRover.id_col
        self.targets = self.dataRover.targets
        self.numericals = self.dataRover.numericals
        self.categoricals = self.dataRover.categoricals        
        
    def cleaningData(self):
#        self.featureMixer = FeatureMixer(self.dataRover)        
        self.features_valid = self.featureMixer.preProcessing()
    
    def mappingTargets(self):
#        self.predictor = Predictor(self.featureMixer)
#        self.predictor.modeling()
        self.predictors = {}
        
        for target in self.targets:
            y = self.train[target]          
            
            if target in self.categoricals:
                noOfClasses = len( y.unique() )
                if noOfClasses > 2:
                    self.predictors[y.name] = predictors.MulticlassClassifier(y)
                else:
                    self.predictors[y.name] = predictors.BinaryClassifier(y, self.featureMixer, self.id_col)
                    
            else:
                self.predictors[y.name] = predictors.Regressor(y, self.featureMixer, self.id_col, self.targets)
    
    def modelTargets(self):
        for target in self.predictors:
            model = self.predictors[target]
            model.model()
            

if __name__ == '__main__':
    rover = datarover.DataRover('/home/abzooba/python-workspace/titanic', 'train.csv', 'test.csv')
    mixer = preprocessing.FeatureMixer(rover)
    
    ml = MachineLearningPipeline(rover, mixer) # ~ KB
    ml.exploringData()
    
    ml.cleaningData()
    ml.mappingTargets()
    ml.modelTargets()
#    ml = DataRover('/home/abzooba/python-workspace/bike-sharing', 'train.csv', 'test.csv') ~ KB
#    ml = DataRover('/home/abzooba/python-workspace/otto', 'train.csv', 'test.csv') # ~ 13 MB
    
#    ml = DataRover('/home/abzooba/python-workspace/rossmann', 'train.csv', 'test.csv') # ~ 40 MB
#    ml = DataRover('/home/abzooba/python-workspace/crime', 'train.csv', 'test.csv') # ~ 130 MB
#    ml = DataRover('/home/abzooba/python-workspace/walmart', 'train.csv', 'test.csv') # ~ 35 MB    
    
    
    
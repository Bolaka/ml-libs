# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:47:38 2015

@author: abzooba
"""

#import oPumpy as np

import re
import imp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

datarover = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/datarover.py')
preprocessing = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/preprocessing.py')
pipeline = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/pipeline.py')
utils = imp.load_source('utils', '/home/abzooba/python-workspace/libs/utils.py')

class TitanicFeatureMixer(preprocessing.FeatureMixer):
    
    def __dictMap__(listOfMajors, non_major):
        mapped_dict = {}
        for i, major in enumerate(reversed(listOfMajors)):
            mapped_dict[major] = (i+1)
        mapped_dict[non_major] = 0
        return mapped_dict
    
    def __dictMap0__(listOfMajors):
        mapped_dict = {}
        for i, major in enumerate(reversed(listOfMajors)):
            mapped_dict[major] = i
        mapped_dict[None] = -99
        return mapped_dict
        
    def __extractName__(self, s):
        rex = re.compile('(?P<lname>\w*),\s*(?P<title>[\w|\s]*).\s*(?P<fname>\w*)\s*(?P<mname>\w*)')   # \s*\((?P<alias>\w*)     
        m = rex.search(s)
        if m is None:
            return s
        else:
            return m.group('title')
    
    def preProcessing(self):
        print 'Titanic feature mixer...'        
        
#        # add the target columns to test data as 9999
#        self.test[self.targets[0]] = "NA"
#        
#        # combine the training and test datasets for data preprocessing
#        combined = pd.concat( [ self.train, self.test ] )          
        
        # Divorce - yes = 2, no = 0, missing = 1
        self.train['Sex'] = self.train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)        
        self.test['Sex'] = self.test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)   
        
        self.train['title'] = self.train['Name'].apply(self.__extractName__)
        self.test['title'] = self.test['Name'].apply(self.__extractName__)
        self.categoricals.append('title')
              
#        self.train['family'] = self.train.SibSp.astype('int') + self.train.Parch.astype('int')
#        self.test['family'] = self.test.SibSp.astype('int') + self.test.Parch.astype('int')
        
#        self.train.loc[self.train.SibSp >= 4, 'SibSp'] = 4     
#        self.test.loc[self.test.SibSp >= 4, 'SibSp'] = 4     
#        self.train.loc[self.train.Parch > 0, 'Parch'] = 1     
#        self.test.loc[self.test.Parch > 0, 'Parch'] = 1
#        self.train.Parch = self.train.Parch.astype('category')
#        self.test.Parch = self.test.Parch.astype('category')
        
        # pre-process categoricals
        for cate in self.categoricals:
            column = self.train[cate]
            dtype = str(column.dtype)
            if dtype == 'object':
                le = LabelEncoder()
                le.fit(np.append(self.train[cate], self.test[cate]))
            
                self.train[cate] = le.transform(self.train[cate])
                self.test[cate] = le.transform(self.test[cate])
                
        self.features = self.categoricals + self.numericals
#        self.features = [ 'title', 'SibSp', 'Parch', 'Pclass', 'Sex', 'Age', 'Survived' ] # 'Fare', 'Embarked'
        
        for t in self.targets:
            self.features.remove(t)
        
        return True

#from pipeline import MachineLearningPipeline
#from datarover import DataRover
#from preprocessing import FeatureMixer
#from predictors import Predictor

rover = datarover.DataRover('/home/abzooba/python-workspace/titanic', 'train.csv', 'test.csv')
mixer = TitanicFeatureMixer(rover)

ml = pipeline.MachineLearningPipeline(rover, mixer) # ~ KB
ml.exploringData(False)

ml.cleaningData()
ml.mappingTargets('error')
ml.modelTargets()
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:27:12 2016

@author: abzooba
"""

import os
import imp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

datarover = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/datarover.py')
preprocessing = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/preprocessing.py')
pipeline = imp.load_source('pipeline', '/home/abzooba/python-workspace/libs/pipeline.py')
utils = imp.load_source('utils', '/home/abzooba/python-workspace/libs/utils.py')


rover = datarover.DataRover('/home/abzooba/python-workspace/otto', 'train.csv', 'test.csv')
mixer = preprocessing.FeatureMixer(rover)

ml = pipeline.MachineLearningPipeline(rover, mixer) # ~ KB
ml.exploringData(True)

ml.cleaningData()
ml.mappingTargets('mlogloss')
ml.modelTargets()

submission = pd.read_csv('/home/abzooba/python-workspace/otto/submission.csv')
dummies = pd.get_dummies(submission['target'])
submission_ = pd.concat([submission, dummies], axis=1)
submission_.drop('target', axis=1, inplace = True)
submission_.to_csv(os.path.join('/home/abzooba/python-workspace/otto/', 'submission_.csv' ), index=False)
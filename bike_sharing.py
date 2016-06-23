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


rover = datarover.DataRover('/home/abzooba/python-workspace/bike-sharing', 'train.csv', 'test.csv')
mixer = preprocessing.FeatureMixer(rover)

ml = pipeline.MachineLearningPipeline(rover, mixer) # ~ KB
ml.exploringData(True)

ml.cleaningData()
ml.mappingTargets('rmsle')
ml.modelTargets()
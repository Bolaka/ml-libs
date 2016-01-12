# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:48:25 2015

@author: abzooba
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

class Classifier(object):
    def __init__(self, y):
        
        self.y = y
        self.classifier = None
        self.noOfClasses = len( y.unique() )
        
        if self.noOfClasses > 2:
            self.classifier = MulticlassClassifier(y)
        else:
            self.classifier = BinaryClassifier(y)
            
    def model(self):
        self.classifier.model()

class Regressor(object):
    
    def __init__(self, y, featureMixer, id_col, targets, eval_metric):
        
        self.directory = featureMixer.directory
        self.id_col = id_col
        self.y = y
        self.train = featureMixer.train
        self.test = featureMixer.test
        self.features = featureMixer.features
        self.targets = targets
        self.eval_metric = eval_metric
        
    def mape( actual, predicted ):
        return np.mean(np.abs((actual - predicted) / actual))
        
    def sle(self, actual, predicted):
        """
        Computes the squared log error.
        This function computes the squared log error between two numbers,
        or for element between a pair of lists or numpy arrays.
        Parameters
        ----------
        actual : int, float, list of numbers, numpy array
                 The ground truth value
        predicted : same type as actual
                    The predicted value
        Returns
        -------
        score : double or list of doubles
                The squared log error between actual and predicted
        """
        _sle_ = np.power( np.log( np.array(actual)+1 ) - np.log( np.array(predicted)+1 ), 2)
        return _sle_

    def se(self, actual, predicted):
        """
        Computes the squared error.
        This function computes the squared error between two numbers,
        or for element between a pair of lists or numpy arrays.
        Parameters
        ----------
        actual : int, float, list of numbers, numpy array
                 The ground truth value
        predicted : same type as actual
                    The predicted value
        Returns
        -------
        score : double or list of doubles
                The squared error between actual and predicted
        """
        return np.power(np.array(actual)-np.array(predicted), 2)
    
    def msle(self, actual, predicted):
        """
        Computes the mean squared log error.
        This function computes the mean squared log error between two lists
        of numbers.
        Parameters
        ----------
        actual : list of numbers, numpy array
                 The ground truth value
        predicted : same type as actual
                    The predicted value
        Returns
        -------
        score : double
                The mean squared log error between actual and predicted
        """
        _msle_ = np.nanmean(self.sle(actual, predicted))    
        return _msle_
    
    def mse(self, actual, predicted):
        """
        Computes the mean squared error.
        This function computes the mean squared error between two lists
        of numbers.
        Parameters
        ----------
        actual : list of numbers, numpy array
                 The ground truth value
        predicted : same type as actual
                    The predicted value
        Returns
        -------
        score : double
                The mean squared error between actual and predicted
        """
        return np.mean(self.se(actual, predicted))
    
    def rmse(self, actual, predicted):
        """
        Computes the root mean squared error.
        This function computes the root mean squared error between two lists
        of numbers.
        Parameters
        ----------
        actual : list of numbers, numpy array
                 The ground truth value
        predicted : same type as actual
                    The predicted value
        Returns
        -------
        score : double
                The root mean squared error between actual and predicted
        """
        return np.sqrt(self.mse(actual, predicted))
    
    def rmsle(self, actual, predicted):
        """
        Computes the root mean squared log error.
        This function computes the root mean squared log error between two lists
        of numbers.
        Parameters
        ----------
        actual : list of numbers, numpy array
                 The ground truth value
        predicted : same type as actual
                    The predicted value
        Returns
        -------
        score : double
                The root mean squared log error between actual and predicted
        """
        return np.sqrt(self.msle(actual, predicted))    
    
    def model(self):
        x = self.train[self.features].values
        x_test = self.test[self.features].values        
        
        # TODO compute from no. of rows & features
        k_folds = 10
        n_times = 1
        
#        dx = xgb.DMatrix(x, label=self.y, missing = float('nan'))        
#        dtest = xgb.DMatrix(x_test, missing = float('nan') )        
        
#        # setup parameters for xgboost
#        param = {}
#        param['objective'] = 'binary:logistic'
#        param['eval_metric'] = 'error'
#        # scale weight of positive examples
#        param['eta'] = 0.3
#        param['max_depth'] = 4
#        param['silent'] = 1
##            param['subsample'] = 0.5
##            param['colsample_bytree'] = 0.6        
        
        scores = []
        iters = []
        
        for n in range(n_times):
#            print '\n==============' + str(n) + '\n'
            skf = StratifiedKFold(self.y, n_folds=k_folds, shuffle=True) 
            
    #        for train_index, validation_index in skf:
            for validation_index, train_index in skf:
                x_train, x_validate = x[train_index], x[validation_index]
                y_train, y_validate = self.y[train_index], self.y[validation_index]        
                
                # Initialize a classifier with key word arguments
                clf = GradientBoostingRegressor()
                clf.fit(x_train,y_train)
                y_predicted = clf.predict(x_validate)
                
                if self.eval_metric == 'rmse':
                    rmse_score = self.rmse(y_validate, y_predicted)
                    scores.append(rmse_score)
                elif self.eval_metric == 'rmsle':
                    rmsle_score = self.rmsle(y_validate, y_predicted)
                    scores.append(rmsle_score)
                    
#                dtrain = xgb.DMatrix(x_train, label=y_train, missing = float('nan') )
#                dval = xgb.DMatrix(x_validate, label=y_validate, missing = float('nan') ) # 
#                
#                watchlist = [ (dtrain,'train'), (dval, 'test') ]
#                num_round = 500
#                clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30, verbose_eval=False) # 
#                scores.append(np.absolute(clf.best_score))
#                iters.append(clf.best_iteration)
        
        print('features used : ' + ' '.join(self.features) )
        print('overall ' + self.eval_metric + ' for ' + str(k_folds) + ' folds = ' + str(np.mean(scores)))

#        print(param)
#        print('XGBoost classifier = ' + str(1-np.mean(scores)) + ' +/- ' + str(round(np.std(scores)*100, 2)) + '%' )
#        n_rounds_max = np.max(iters) + 1
#        
##        xgb_clf = xgb.train(param, dx, n_rounds_max)
#        self.predictions = xgb_clf.predict(dtest)
#        
#        self.predictions[self.predictions >= 0.5] = 1
#        self.predictions[self.predictions < 0.5] = 0
#        
#        submission = pd.DataFrame({ self.id_col: self.test['PassengerId'],
#                            self.y.name: self.predictions })
#        submission[self.y.name] = submission[self.y.name].astype(int)
#        submission.to_csv(os.path.join(self.directory, 'submission.csv' ), index=False)
        
        
class BinaryClassifier(Classifier):
    
    def __init__(self, y, featureMixer, id_col):
        
        self.directory = featureMixer.directory
        self.id_col = id_col
        self.y = y
        self.train = featureMixer.train
        self.test = featureMixer.test
        self.features = featureMixer.features
        
    def model(self):
        
        x = self.train[self.features].values
        x_test = self.test[self.features].values        
        
        # TODO compute from no. of rows & features
        k_folds = 10
        n_times = 1
        
        dx = xgb.DMatrix(x, label=self.y, missing = float('nan'))        
        dtest = xgb.DMatrix(x_test, missing = float('nan') )        
        
        # setup parameters for xgboost
        param = {}
        param['objective'] = 'binary:logistic'
        param['eval_metric'] = 'error'
        # scale weight of positive examples
        param['eta'] = 0.3
        param['max_depth'] = 4
        param['silent'] = 1
#            param['subsample'] = 0.5
#            param['colsample_bytree'] = 0.6        
        
        scores = []
        iters = []
        
        for n in range(n_times):
#            print '\n==============' + str(n) + '\n'
            skf = StratifiedKFold(self.y, n_folds=k_folds, shuffle=True) 
            
    #        for train_index, validation_index in skf:
            for validation_index, train_index in skf:
                x_train, x_validate = x[train_index], x[validation_index]
                y_train, y_validate = self.y[train_index], self.y[validation_index]        
                
                dtrain = xgb.DMatrix(x_train, label=y_train, missing = float('nan') )
                dval = xgb.DMatrix(x_validate, label=y_validate, missing = float('nan') ) # 
                
                watchlist = [ (dtrain,'train'), (dval, 'test') ]
                num_round = 500
                clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30, verbose_eval=False) # 
                scores.append(np.absolute(clf.best_score))
                iters.append(clf.best_iteration)
        
        print('\n=========== overall eval metric for ' + str(k_folds) + ' folds ===========')      

        print(' '.join(self.features) + '\n')
        print(param)
        print('XGBoost classifier = ' + str(1-np.mean(scores)) + ' +/- ' + str(round(np.std(scores)*100, 2)) + '%' )
        n_rounds_max = np.max(iters) + 1
        
        xgb_clf = xgb.train(param, dx, n_rounds_max)
        self.predictions = xgb_clf.predict(dtest)
        
        self.predictions[self.predictions >= 0.5] = 1
        self.predictions[self.predictions < 0.5] = 0
        
        submission = pd.DataFrame({ self.id_col: self.test[self.id_col],
                            self.y.name: self.predictions })
        submission[self.y.name] = submission[self.y.name].astype(int)
        submission.to_csv(os.path.join(self.directory, 'submission.csv' ), index=False)

class MulticlassClassifier(Classifier):

    def __init__(self, y, featureMixer, id_col):
        self.directory = featureMixer.directory
        self.id_col = id_col
        self.y = y

        dtype = y.dtype.name
        if dtype == 'object':
            print y.name + ' is being encoded...'
            le = LabelEncoder()
            self.y = le.fit_transform(y.values)        
        
        self.noOfClasses = len( y.unique() )
        self.train = featureMixer.train
        self.test = featureMixer.test
        self.features = featureMixer.features
        
    def model(self):
        x = self.train[self.features].values
        x_test = self.test[self.features].values        
        
        # TODO compute from no. of rows & features
        k_folds = 10
        n_times = 1
        
        dx = xgb.DMatrix(x, label=self.y, missing = float('nan'))        
        dtest = xgb.DMatrix(x_test, missing = float('nan') )        
        
        # setup parameters for xgboost
        param = {}
        param['objective'] = 'multi:softmax'
        param['eval_metric'] = 'mlogloss'
        param['num_class'] = self.noOfClasses
        param['eta'] = 0.3
        param['max_depth'] = 4
        param['silent'] = 1
#            param['subsample'] = 0.5
#            param['colsample_bytree'] = 0.6        
        
        scores = []
        iters = []
        
        for n in range(n_times):
            print '---------------- ' + str(n+1)
            skf = StratifiedKFold(self.y, n_folds=k_folds, shuffle=True) 
            
    #        for train_index, validation_index in skf:
            for validation_index, train_index in skf:
                x_train, x_validate = x[train_index], x[validation_index]
                y_train, y_validate = self.y[train_index], self.y[validation_index]        
                
                dtrain = xgb.DMatrix(x_train, label=y_train, missing = float('nan') )
                dval = xgb.DMatrix(x_validate, label=y_validate, missing = float('nan') ) # 
                
                watchlist = [ (dtrain,'train'), (dval, 'test') ]
                num_round = 500
                clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30, verbose_eval=False) # 
                scores.append(np.absolute(clf.best_score))
                iters.append(clf.best_iteration)
        
        print('\n=========== overall eval metric for ' + str(k_folds) + ' folds ===========')      

        print(' '.join(self.features) + '\n')
        print(param)
        print('XGBoost classifier = ' + str(1-np.mean(scores)) + ' +/- ' + str(round(np.std(scores)*100, 2)) + '%' )
        n_rounds_max = np.max(iters) + 1
        
        xgb_clf = xgb.train(param, dx, n_rounds_max)
        self.predictions = xgb_clf.predict(dtest)
        print self.predictions
        
        submission = pd.DataFrame({ self.id_col: self.test[self.id_col],
                            self.y.name: self.predictions })
        submission[self.y.name] = submission[self.y.name]
        submission.to_csv(os.path.join(self.directory, 'submission.csv' ), index=False)

class Predictor(object):
    
    def __init__(self, featureMixer):
        self.predictors = {}
        
        self.targets = featureMixer.targets
        self.train = featureMixer.train
        self.test = featureMixer.test
        self.categoricals = featureMixer.categoricals
        self.numericals = featureMixer.numericals
        
        for target in self.targets:
            y = self.train[target]          
                
            if target in self.categoricals:
                self.predictors[y.name] = Classifier(y)
#                self.__classification__(y)
            else:
#                self.__regression__()
                self.predictors[y.name] = Regressor()
    
    def computeFolds(self, nrows, nfeats):
        pass
    
    def __regression__(self):
        # TODO a regression
        print 'YO!!! regression'    
    
    def modeling(self):
        for target, modeler in self.predictors:
            print 'modeling ' + target
            modeler.model()
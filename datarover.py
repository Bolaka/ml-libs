# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:45:25 2015

@author: abzooba
"""
from __future__ import division
import imp
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#from numpy.random import seed

utils = imp.load_source('utils', '/home/abzooba/python-workspace/libs/utils.py')

# suppress pandas warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

sns.set(style="whitegrid", color_codes=True)
#sns.set(rc={"figure.figsize": (6, 6)})

class DataRover(object):
    
    """Surf data with this interface. Data Surfers have the
    following properties:

    Attributes:
        directory: A string representing the data source folder
        train: A pandas dataframe consisting of the training data
        test: A pandas dataframe consisting of the testing data
        train_cols: Column names on the training data
        test_cols: Column names on the testing data
        target: The target column
        id_cols: ID columns
        sparse_cols: Sparse column names
        sparse_rows: Sparse row names
        outlier_cols: Column names which have outliers
        categoricals: The categorical columns
        numericals: The numerical columns
        date_cols: Date columns
    """    

    univariate_dir = 'univariate'
    bivariate_dir = 'bivariate'
    metafile = 'metadata.csv'    
    sparse_limit = 50
    outlier_limit = 5
    distinct_cutoff = 10
    
    def __init__(self, directory, trainfile, testfile):
        """The constructor
        """
        print 'in data rover... exploring ' + directory
#        print type(train)        
        
        # the data source
        self.directory = directory
        trainfile = os.path.join( directory ,trainfile )
        testfile = os.path.join( directory , testfile )
        
        # TODO - handle JSON data
        self.train = pd.read_csv(trainfile)
        self.test = pd.read_csv(testfile)

        self.memory_usage = self.train.memory_usage(index=True).sum() + self.test.memory_usage(index=True).sum()
        
        print 'train data : ' + str(self.train.shape[0]) + ' rows, ' + str(self.train.shape[1]) + ' columns'                
        print 'test data : ' + str(self.test.shape[0]) + ' rows, ' + str(self.test.shape[1]) + ' columns'
        print 'total memory usage : ' + str(self.memory_usage/1024.0) + ' (KB)'
        
        self.train_cols = self.train.columns
        self.test_cols = self.test.columns
        self.targets = list(set(self.train_cols) - set(self.test_cols))
        print '\nTargets:'
        print ', '.join(self.targets)
        
        self.id_col = None
        self.id_cols = []
        self.sparse_rows = []
        self.sparse_cols = []
        self.outlier_cols = []
        self.categoricals = []
        self.numericals = []
        
        # TODO detect date columns
        self.date_cols = []
    
    def __addMissing__(self, s):
        """Internal method to calculate percentage missing for given vector
        """
        return ( float(s.isnull().sum()) / len(s) ) * 100
    
    def extractMetadata(self, setID = True):
        """Extracts metadata from data
        """
        missingness = self.train.apply(self.__addMissing__, 1)    
        self.sparse_rows = missingness[missingness>self.sparse_limit].index.values
        
        # we will calculate these meta columns...
        meta_columns = ['type', 'missing %', 'variance', 'distinct %', 'mode', 'mean', 'median', 
                        'min', 'max', 'STD', 'distanceSTD_mean']
        self.metadata = pd.DataFrame(index=self.train_cols, columns=meta_columns)
        
        # ... for each column one by one
        for colname in self.train_cols:
            column = self.train[colname]
            dtype = str(column.dtype)
            distinct = ( len(column.unique()) / len(column) ) * 100
            
            # id check
            if distinct == 100:
                self.id_cols.append(colname)
            # type check!
            elif dtype != 'object':
                # checking integer columns for categorical type by fitting a linear model
                # and checking R-squared 
                y = np.sort(column.unique())            
                x = range(1, len(y) + 1)
                gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#                print colname, r_value
                if r_value == 1.0 and len(y) < len(column)/2:
                    # this column qualifies as a categorical feature!
                    self.train[colname] = self.train[colname].astype('category', ordered=True)
                    
                    if colname not in self.targets:
                        self.test[colname] = self.test[colname].astype('category', ordered=True)
                                        
                    # re-initializations
                    column = self.train[colname]
                    dtype = str(column.dtype)
            
            missing = ( column.isnull().sum() / len(column) ) * 100
            variance = float('nan')
            mode = float('nan')
            mean = float('nan')
            median = float('nan')
            mini = float('nan')
            maxi = float('nan')
            std = float('nan')
            distanceSTD_mean = float('nan')
            
            # numeric
            if dtype != 'object' and dtype != 'category':
                variance = column.var()
                mean = column.mean()
                median = column.median()
                mini = column.min()
                maxi = column.max()
                std = column.std()
                
                distanceSTD_mean = max(np.abs( maxi - mean )/std, np.abs( mini - mean )/std)
                
                if distanceSTD_mean > self.outlier_limit:
                    self.outlier_cols.append(colname)
                    
                self.numericals.append(colname)
            elif len(column.mode()) > 0 and distinct < self.distinct_cutoff:
                mode = ', '.join( str(x) for x in column.mode().values )
                
#                # this column qualifies as a categorical feature!
#                self.train[colname] = self.train[colname].astype('category')
#                if colname not in self.targets:
#                    self.test[colname] = self.test[colname].astype('category')      
                
#                column = self.train[colname]
#                dtype = str(column.dtype)
                self.categoricals.append(colname)
            
            self.metadata.loc[colname] = pd.Series({
                                        'type':dtype, 
                                        'missing %':missing, 
                                        'variance':variance, 
                                        'distinct %':distinct,
                                        'mode':mode, 
                                        'mean':mean, 
                                        'median':median, 
                                        'min':mini, 
                                        'max':maxi, 
                                        'STD':std, 
                                        'distanceSTD_mean':distanceSTD_mean }) 
            if missing > self.sparse_limit:
                self.sparse_cols.append(colname)
        
        self.metadata.to_csv(os.path.join(self.directory, self.metafile )) # 
        
        if len(self.id_cols) > 0:
            print '\nID column probables:'
            print ', '.join(self.id_cols)   
            
            self.id_col = self.id_cols[0]
            if setID == True and len(self.id_cols) > 1:
                self.id_col = utils.getInput('Please choose an ID: ', self.id_cols)
            
            if self.id_col in self.numericals:
                self.numericals.remove(self.id_col)
            elif self.id_col in self.categoricals:
                self.categoricals.remove(self.id_col)
#        else:
#            # TODO properly...
#            options = [x for x in self.train_cols if x not in self.targets]
#            ids = utils.getInputs('Oops, no ID? Choose columns (comma separated) to be used as ID (' + ', '.join(options) + '):', options)
#            print ids
#            idName = utils.getInput('Care to give a name for the ID: ', [])
#            print idName
#            
#            self.train[idName] = self.train[ids[0]].map(str) + "__" + self.train[ids[1]]
#            self.test[idName] = self.test[ids[0]].map(str) + "__" + self.test[ids[1]].map(str)
#            self.id_cols.append(idName)
            
        if len(self.numericals) > 0:
            print '\nNumerical columns:'
            print ', '.join( self.numericals )        
        
        if len(self.categoricals) > 0:
            print '\nCategorical columns:'
            print ', '.join( self.categoricals )         
        
        if len(self.sparse_cols) > 0:
            print '\nSparse columns:'
            print ', '.join( self.sparse_cols )     

        if len(self.sparse_rows) > 0:
            print '\nSparse rows:'    
            print ', '.join( self.sparse_rows )    
        
        if len(self.outlier_cols) > 0:
            print '\nOutlier columns:'
            print ', '.join( self.outlier_cols )     
        
        print '\nSummary statistics:'
        print '========================================================================================================================================'
        with pd.option_context('display.max_rows', 999, 'display.max_columns', 11, 'display.precision', 2, 'display.width', 1000):
            print self.metadata
        print '========================================================================================================================================'
    
    def univariate(self):
        path_to_uni_plots = os.path.join(self.directory, self.univariate_dir)
        if not os.path.exists(path_to_uni_plots):
            os.makedirs(path_to_uni_plots)        
        
        for num_col in self.numericals:
            x = self.train[num_col].dropna()
            plt.figure(figsize=(10, 8))
            sns.distplot(x, kde=False, rug=False)
            plt.savefig(os.path.join(path_to_uni_plots, num_col + '_histogram.png' ), bbox_inches='tight')  
            
        for cat_col in self.categoricals:
            if len( self.train[cat_col].unique() ) < 10:
                x = self.train[cat_col].dropna()
                plt.figure(figsize=(10, 8))
                sns.countplot(x=cat_col, data=self.train)
                plt.savefig(os.path.join(path_to_uni_plots, cat_col + '_bar.png' ), bbox_inches='tight')  
    
    def __categorical_numerical__(self, x, y, path_to_bi_plots):
        print x.name + ' vs ' + y.name        
        sns.boxplot(x=x, y=y)
        plt.savefig(os.path.join(path_to_bi_plots, x.name + '_' + y.name + '_boxplot.png' ), bbox_inches='tight')  
    
    def __numerical_numerical__(self, x, y, path_to_bi_plots):
#        print x.name + ' vs ' + y.name        
        sns.jointplot(x, y, kind="reg") # , kind="kde"
        plt.savefig(os.path.join(path_to_bi_plots, x.name + '_' + y.name + '_scatter.png' ), bbox_inches='tight')  
     
    def __categorical_categorical__(self, x, y, path_to_bi_plots):
#        print x.name + ' v ' + y.name     
        freq_table = pd.crosstab(index=y, columns=x)
#        print freq_table
        sns.heatmap(freq_table, annot=True, fmt="d", cbar=False, square=True, linewidths=.5)
#        pd.pivot_table(ml.train, values="Age", index=["Pclass"],columns=["Sex"])
#        sns.jointplot(x, y) # , kind="kde"
        plt.savefig(os.path.join(path_to_bi_plots, x.name + '_' + y.name + '_heatmap.png' ), bbox_inches='tight')  
      
    def bivariateTarget(self):
        path_to_bi_plots = os.path.join(self.directory, self.bivariate_dir)
        if not os.path.exists(path_to_bi_plots):
            os.makedirs(path_to_bi_plots)
         
        for target in self.targets:
            y = self.train[target] #.dropna()            
            
            for feature in self.test_cols:
                plt.figure(figsize=(10, 8))
                x = self.train[feature] #.dropna()                
                
                if target in self.categoricals and feature in self.categoricals:
#                    print target_type + ' : ' + feature_type
                    self.__categorical_categorical__(x, y, path_to_bi_plots)
                elif target in self.categoricals and feature in self.numericals:
                    self.__categorical_numerical__(y, x, path_to_bi_plots)
                elif target in self.numericals and feature in self.numericals:
#                    print target + ' vs ' + feature
                    self.__numerical_numerical__(x, y, path_to_bi_plots)
                elif target in self.numericals and feature in self.categoricals:
                    self.__categorical_numerical__(x, y, path_to_bi_plots)
         
    def bivariate(self, xname, yname):
        path_to_bi_plots = os.path.join(self.directory, self.bivariate_dir)
        x = self.train[xname]
        y = self.train[yname]
        if xname in self.categoricals and yname in self.categoricals:
            self.__categorical_categorical__(x, y, path_to_bi_plots)
        elif xname in self.categoricals and yname in self.numericals:
            self.__categorical_numerical__(x, y, path_to_bi_plots)
        elif xname in self.numericals and yname in self.numericals:
            self.__numerical_numerical__(x, y, path_to_bi_plots)
        elif xname in self.numericals and yname in self.categoricals:
            self.__categorical_numerical__(y, x, path_to_bi_plots)
    
#    def multivariate(self, xname, yname, byname):
#        path_to_bi_plots = os.path.join(self.directory, self.bivariate_dir)
#        x = self.train[xname]
#        y = self.train[yname]
#        by = self.train[byname]        
#        
#        if xname in self.numericals and yname in self.numericals and byname in self.categoricals: 
            
    
    @staticmethod
    def static():
        print 'machine learning!'
        
if __name__ == '__main__':
    ml = DataRover('/home/abzooba/python-workspace/titanic', 'train.csv', 'test.csv') # ~ KB
#    ml = DataRover('/home/abzooba/python-workspace/bike-sharing', 'train.csv', 'test.csv') ~ KB
#    ml = DataRover('/home/abzooba/python-workspace/otto', 'train.csv', 'test.csv') # ~ 13 MB
    
#    ml = DataRover('/home/abzooba/python-workspace/rossmann', 'train.csv', 'test.csv') # ~ 40 MB
#    ml = DataRover('/home/abzooba/python-workspace/crime', 'train.csv', 'test.csv') # ~ 130 MB
#    ml = DataRover('/home/abzooba/python-workspace/walmart', 'train.csv', 'test.csv') # ~ 35 MB    
    
    valid = ml.extractMetadata()
    
    if valid != False:
        ml.univariate()
        ml.bivariateTarget()
    #    ml.bivariate('season', 'weather')
    #    ml.bivariate('season', 'temp')
    #    ml.bivariate('temp', 'atemp')
        ml.preProcessing()
        ml.modeling()
    
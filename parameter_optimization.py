from direct_optimisation import main_classifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform
from scipy.stats import expon
import pickle
import numpy as np
from fashion_mnist import mnist_reader
import vectorisation as vect
import pandas as pd
from auxiliary_functions import *
from numpy.random import seed


def parameter_optimization(train_index, y_train, vectorisation_methods, feature_dictionary,search_method, normalization):
    func_list = [getattr(vect, keys) for keys in vectorisation_methods.keys()]
    best_scores = dict()
    for func in func_list:
        func_parameters =load_parameters(func,vectorisation_methods)
        for p in func_parameters:
            X_train, y_train = build_dataset_from_features(train_index,y_train,func,feature_dictionary,p)
            X_train = np.array(X_train)
            if normalization:             
                mm_scaler = MinMaxScaler()
                X_train = mm_scaler.fit_transform(X_train)
            search_method.fit(X_train, y_train)
            best_scores[func.__name__+'_'+str(p)] = (search_method.best_params_, search_method.best_score_)
    return best_scores
            
            
    
    
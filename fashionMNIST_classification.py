import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from fashionMNIST_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
from fashion_mnist import mnist_reader
             
def classification(func, str_p='', str_q='', base_estimator='RF', 
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3):
    
    if str_p!='':
        p = str_p
        str_p = '_'+str_p
    if str_q!='':
        str_q = '_'+str_q
            
    #perform the classification using the direct_optimization method for all
    #functions except for TropicalCoordinates, where tropical_optimization is
    #used instead

    path_feat = "fashion_mnist/features/"
    path_diag= "fashion_mnist/pdiagrams/"
    

    _, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
    _, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

    n_train = len(y_train)
    n_total = len(y_train) + len(y_test)

    Z_train = range(n_train)
    Z_test = range(n_train, n_total)
    
    if func==GetPersTropicalCoordinatesFeature:
        method = tropical_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree, r=float(p))
        
            
        X_train, X_test = Z_train, Z_test
    else: 
        with open(path_feat + func.__name__ + '_l.pkl', 'rb') as f:
            features_l = pickle.load(f)
        with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
            features_u = pickle.load(f)
        
    
        
        method = main_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree)
        
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
                        features_l[str(i) + str_p + str_q],
                        features_u[str(i) + str_p + str_q]
                    ]
                    ))
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l[str(i) + str_p + str_q],
                        features_u[str(i) + str_p + str_q]
                    ]
                    ))    
    
    X_train = float64to32(X_train)
    X_test = float64to32(X_test)
    
    score_list = []
    for i in range(100):
        method.fit(X_train, y_train)
        score_list.append(np.mean(y_test.ravel() == method.predict(X_test)))
        
    print(np.mean(score_list))
    

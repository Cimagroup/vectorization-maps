import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from OUTEX_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
from numpy.random import choice
from numpy.random import seed


def classification(func, str_p='', str_q='', base_estimator='RF', 
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3, 
             s=1, rs=0):
    
    if str_p!='':
        p = str_p
        str_p = '_'+str_p
    if str_q!='':
        str_q = '_'+str_q
            
    #perform the classification using the direct_optimization method for all
    #functions except for TropicalCoordinates, where tropical_optimization is
    #used instead
    
    path_feat = 'Outex-TC-00024/features/' 
    path_data = 'Outex-TC-00024/data/000/'
    
    seed(s)
    labels = range(68)
    labels = choice(labels, size=(10), replace = False)

    train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                        usecols=[1]).to_numpy().flatten().tolist())
    test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                       usecols=[1]).to_numpy().flatten().tolist())
    train_indexes = np.array([i for i in range(len(train_labels)) if train_labels[i] in labels])
    test_indexes = np.array([i for i in range(len(test_labels)) if test_labels[i] in labels])
    label_list = np.hstack([train_labels[train_indexes], test_labels[test_indexes]])


    Z_train, Z_test, y_train, y_test = train_test_split(range(len(label_list)), 
                                                        label_list, test_size=0.3, 
                                                        random_state=rs)
    if func!=GetPersTropicalCoordinatesFeature:
        with open(path_feat + func.__name__ + '_l_d0.pkl', 'rb') as f:
            features_l_d0 = pickle.load(f)
        with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
            features_l_d1 = pickle.load(f)
        with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
            features_u_d0 = pickle.load(f)
        with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
            features_u_d1 = pickle.load(f)
        
    
        
        method = main_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree)
        
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
                        features_l_d0[str(i) + str_p + str_q],
                        features_l_d1[str(i) + str_p + str_q],
                        features_u_d0[str(i) + str_p + str_q],
                        features_u_d1[str(i) + str_p + str_q]
                    ]
                    ))
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l_d0[str(i) + str_p + str_q],
                        features_l_d1[str(i) + str_p + str_q],
                        features_u_d0[str(i) + str_p + str_q],
                        features_u_d1[str(i) + str_p + str_q]
                    ]
                    ))    
    
        X_train = float64to32(X_train)
        X_test = float64to32(X_test)
    
    else:
        method = tropical_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree, r=float(p))
        
            
        X_train, X_test = Z_train, Z_test
    
    score_list = []
    for i in range(10):
        method.fit(X_train, y_train)
        y_pred = method.predict(X_test)
        score_list.append(np.mean(y_test.ravel() == method.predict(X_test)))

    return np.mean(score_list)
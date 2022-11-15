import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from direct_optimisation import main_classifier
from SHREC14_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
from numpy.random import choice
from numpy.random import seed
import copy

#%%
def classification(func, t, str_p='', str_q='', base_estimator='RF', 
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3, 
             s=1, rs=0, iters=100):
    
    feat_path = "Shrec14/features/"
    seed(1)
    
    #extract the parameters of the method
    if str_p!='':
        if np.char.isdigit(str_p):
            p = int(str_p)
        elif np.char.isdigit(str_p.replace('.','',1)):
            p = float(str_p)
        else:
            p = copy.deepcopy(str_p)
            str_p = '_'+str_p
    if str_q!='':
        if np.char.isdigit(str_q):
            q = int(str_q)
        elif np.char.isdigit(str_q.replace('.','',1)):
            q = float(str_q)
        else:
            q = copy.deepcopy(str_q)
            str_q = '_'+str_q
        
    #Load the labels
    with open(feat_path + 'Z_train' +'.pkl', 'rb') as f:
        Z_train = pickle.load(f)
    with open(feat_path + 'Z_test' +'.pkl', 'rb') as f:
        Z_test = pickle.load(f)
    with open(feat_path + 'y_train' +'.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(feat_path + 'y_test' +'.pkl', 'rb') as f:
        y_test = pickle.load(f)
        
    if func==GetPersTropicalCoordinatesFeature:
        with open(feat_path + 'dgmsT' +'.pkl', 'rb') as f:
            dgmsT = pickle.load(f)
        method = tropical_classifier(base_estimator=base_estimator,  t=t, 
                     n_estimators=n_estimators, C=C, gamma=gamma, r=p, 
                     degree=degree, kernel=kernel,  dgmsT=dgmsT)
        
        X_train, X_test = Z_train[str(t)], Z_test[str(t)]
    else: 
        with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
            features = pickle.load(f)
        X_train = []
        for i in Z_train[str(t)]:
            if str_q!='':
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
            elif str_p!='':
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
            else:
                X_train.append(features[str(t)+'_'+str(i)])
        X_test = []
        for i in Z_test[str(t)]:
            if str_q!='':
                X_test.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
            elif str_p!='':
                X_test.append(features[str(t)+'_'+str(p)+'_'+str(i)])
            else:
                X_test.append(features[str(t)+'_'+str(i)])
        
    
        
        method = main_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree)
        mm_scaler = MinMaxScaler()
        X_train = mm_scaler.fit_transform(X_train)
        X_test = mm_scaler.transform(X_test)
    
    y_train = np.array(y_train[str(t)])
    y_test = np.array(y_test[str(t)])
    score_list = []
    for i in range(iters):
        method.fit(X_train, y_train)
        y_pred = method.predict(X_test)
        score_list.append(np.mean(y_test.ravel() == method.predict(X_test)))

    return np.mean(score_list)

#%%
def best_parameter(features, t):
    features = features[str(t)]
    n_estimators = 100
    C = 1
    kernel = 'rbf'
    gamma = 0.1
    degree = 3
    if type(features)==tuple:
        feat = features[0]
        parameters = ""
            
    elif type(features)==dict:
        keys = features.keys()
        index = np.argmax([features[key][1] for key in keys])
        keys_l=list(keys)
        parameters = keys_l[index]
        feat = features[keys_l[index]][0]
    
    estimator = feat["base_estimator"]
    estimator_params = feat.keys()
        
    if 'n_estimators' in estimator_params:
        n_estimators = feat["n_estimators"]
    if 'C' in estimator_params:
        C = feat["C"]
    if 'kernel' in estimator_params:
        kernel = feat["kernel"]
    if 'gamma' in estimator_params:
        gamma = feat["gamma"]
    if 'degree' in estimator_params:
        degree = feat["degree"]
    if 'r' in estimator_params:
        parameters = str(feat["r"])
        
    return (parameters, estimator, n_estimators, C, kernel, gamma, degree)


path_feat = "Shrec14/features/"
func_list = [
             GetPersStats,
             GetCarlssonCoordinatesFeature,
             GetPersEntropyFeature,
             GetBettiCurveFeature,
             GetTopologicalVectorFeature,
             GetPersLifespanFeature,
             GetAtolFeature,
             GetPersImageFeature, 
             GetPersSilhouetteFeature,
             GetComplexPolynomialFeature,
             GetPersLandscapeFeature,
             GetTemplateFunctionFeature,
             GetAdaptativeSystemFeature,
             GetPersTropicalCoordinatesFeature
            ]

for func in func_list:
    with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'rb') as f:
            features = pickle.load(f)
            
    for t in range(1,11):
        if ((func != GetAtolFeature) and (func != GetTemplateFunctionFeature)) or (t<10):
            parameters, estimator, n_estimators, C, kernel, gamma, degree = best_parameter(features, t=t)
            parameters=parameters.split("_")
            
            if parameters[0]=='':
                result=classification(func, t=t, str_p="", str_q="", base_estimator=estimator, 
                         n_estimators=n_estimators, C=C, kernel=kernel, gamma=gamma, 
                         degree=degree, iters=100)    
            elif len(parameters)==1:
                result=classification(func, t=t, str_p=parameters[0], str_q="", base_estimator=estimator, 
                         n_estimators=n_estimators, C=C, kernel=kernel, gamma=gamma, 
                         degree=degree, iters=100)     
            elif len(parameters)==2:
                result=classification(func, t=t, str_p=parameters[0], str_q=parameters[1], 
                               base_estimator=estimator, n_estimators=n_estimators,
                               C=C, kernel=kernel,gamma=gamma, degree=degree, iters=100)
        
            print(func.__name__)
            print('t=', t)
            print(result)
            print()
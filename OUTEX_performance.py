import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from direct_optimisation import main_classifier
from OUTEX_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
from numpy.random import choice
from numpy.random import seed

#%%
def classification(func, str_p='', str_q='', base_estimator='RF', 
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3, 
             s=1, rs=0, iters=100, n=10):
    
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
    
    # Fix the labels, train and test.
    if n==10:
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
    elif n==68:
        train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                            usecols=[1]).to_numpy().flatten().tolist())
        test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                           usecols=[1]).to_numpy().flatten().tolist())
        labels = np.hstack([train_labels, test_labels])
        Z_train, Z_test, y_train, y_test = train_test_split(range(2720), labels, 
                                                            test_size=0.3, 
                                                            random_state=rs)

    if func==GetPersTropicalCoordinatesFeature:
        method = tropical_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree, r=float(p))
        
            
        X_train, X_test = Z_train, Z_test
    else: 
        if (func == GetTemplateFunctionFeature) or (func == GetAdaptativeSystemFeature):
            with open(path_feat + func.__name__ + str(n) + '_l_d0.pkl', 'rb') as f:
                features_l_d0 = pickle.load(f)
            with open(path_feat + func.__name__ + str(n) + '_l_d1.pkl', 'rb') as f:
                features_l_d1 = pickle.load(f)
            with open(path_feat + func.__name__ + str(n) + '_u_d0.pkl', 'rb') as f:
                features_u_d0 = pickle.load(f)
            with open(path_feat + func.__name__ + str(n) + '_u_d1.pkl', 'rb') as f:
                features_u_d1 = pickle.load(f)
        else:
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
    
        mm_scaler = MinMaxScaler()
        X_train = mm_scaler.fit_transform(X_train)
        X_test = mm_scaler.transform(X_test)
    
    score_list = []
    for i in range(iters):
        method.fit(X_train, y_train)
        y_pred = method.predict(X_test)
        score_list.append(np.mean(y_test.ravel() == method.predict(X_test)))

    return np.mean(score_list)

#%%
def best_parameter(features):
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

#%%
print(10)
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

path_feat = 'Outex-TC-00024/features/' 

for func in func_list:
    with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'rb') as f:
            features = pickle.load(f)
    
    parameters, estimator, n_estimators, C, kernel, gamma, degree = best_parameter(features)
    parameters=parameters.split("_")
    
    if parameters[0]=='':
        result=classification(func, str_p="", str_q="", base_estimator=estimator, 
                 n_estimators=n_estimators, C=C, kernel=kernel, gamma=gamma, 
                 degree=degree, iters=100, n=10)    
    elif len(parameters)==1:
        result=classification(func, str_p=parameters[0], str_q="", base_estimator=estimator, 
                 n_estimators=n_estimators, C=C, kernel=kernel, gamma=gamma, 
                 degree=degree, iters=100, n=10)     
    elif len(parameters)==2:
        result=classification(func, str_p=parameters[0], str_q=parameters[1], 
                       base_estimator=estimator, n_estimators=n_estimators,
                       C=C, kernel=kernel,gamma=gamma, degree=degree, iters=100, n=10)
        
    print(func.__name__)
    print(result)
    print()
    
#%%
print()
print(68)
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

path_feat = 'Outex-TC-00024/features/' 

for func in func_list:
    with open(path_feat + func.__name__ + '_68_hyperparameter.pkl', 'rb') as f:
            features = pickle.load(f)
    
    parameters, estimator, n_estimators, C, kernel, gamma, degree = best_parameter(features)
    parameters=parameters.split("_")
    
    if parameters[0]=='':
        result=classification(func, str_p="", str_q="", base_estimator=estimator, 
                 n_estimators=n_estimators, C=C, kernel=kernel, gamma=gamma, 
                 degree=degree, iters=100, n=68)    
    elif len(parameters)==1:
        result=classification(func, str_p=parameters[0], str_q="", base_estimator=estimator, 
                 n_estimators=n_estimators, C=C, kernel=kernel, gamma=gamma, 
                 degree=degree, iters=100, n=68)     
    elif len(parameters)==2:
        result=classification(func, str_p=parameters[0], str_q=parameters[1], 
                       base_estimator=estimator, n_estimators=n_estimators,
                       C=C, kernel=kernel,gamma=gamma, degree=degree, iters=100, n=68)
        
    print(func.__name__)
    print(result)
    print()
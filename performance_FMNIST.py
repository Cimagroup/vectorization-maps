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
import fashionMNIST_classification as cl
import fashionMNIST_tropical_optimisation as TO

def best_parameter(features):
    keys = features.keys()
    index = np.argmax([features[key][1] for key in keys])
    keys_l=list(keys)
    parameters = keys_l[index]
    estimator = features[keys_l[index]][0]["base_estimator"]
    n_estimators = features[keys_l[index]][0]["n_estimators"]
    return (parameters, estimator, n_estimators)


func_list = [
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
path_feat = "fashion_mnist/features/"
path_diag= "fashion_mnist/pdiagrams/"



func = func_list[0]
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'rb') as f:
            features_u = pickle.load(f)

r = features_u[0]['r']

cl.classification(func, str_p=str(r), str_q='', base_estimator='RF', 
             n_estimators=500, C=1.0, kernel='rbf', gamma=0.1, degree=3,iterations=100)



"""
cl.classification(GetPersStats, str_p='', str_q='', base_estimator='RF', 
             n_estimators=500, C=1.0, kernel='rbf', gamma=0.1, degree=3,iterations=100)
cl.classification(GetCarlssonCoordinatesFeature, str_p='', str_q='', base_estimator='RF', 
             n_estimators=500, C=1.0, kernel='rbf', gamma=0.1, degree=3,iterations=100)

for func in func_list:
    with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'rb') as f:
            features_u = pickle.load(f)
    
    parameters, estimator, n_estimators = best_parameter(features_u)
    parameters=parameters.split("_")
    if len(parameters)==1:
        cl.classification(func, str_p=parameters[0], str_q="", base_estimator=estimator, 
                 n_estimators=n_estimators, C=1.0, kernel='rbf', gamma=0.1, degree=3,iterations=100)    
    if len(parameters)==2:
        cl.classification(func, str_p=parameters[0], str_q=parameters[1], base_estimator=estimator, 
                 n_estimators=n_estimators, C=1.0, kernel='rbf', gamma=0.1, degree=3,iterations=100)    
"""                 
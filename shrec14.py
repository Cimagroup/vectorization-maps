from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from scipy.stats import expon
import pickle
import numpy as np
from feature_computation import *
import vectorisation as vect
import pandas as pd
from direct_optimisation import main_classifier
from auxiliary_functions import *
from numpy.random import seed

s=1
seed(s)

#%%
vec_methods = dict()
vec_methods['GetPersStats']=(),
# vec_methods['GetCarlssonCoordinatesFeature']=(),
# vec_methods['GetPersEntropyFeature'] = [[50,100,200]]
# vec_methods['GetBettiCurveFeature'] = [[50,100,200]]
# vec_methods['GetPersLifespanFeature'] = [[50,100,200]]
# vec_methods['GetAtolFeature'] = [[2,4,8,16]]
# vec_methods['GetPersTropicalCoordinatesFeature'] = [[10,50,250,500,800]]
# vec_methods['GetPersImageFeature'] = [[0.001,0.01,0.2,1],[10,20,50]]
# vec_methods['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
# vec_methods['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
# vec_methods['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]
# vec_methods['GetTemplateFunctionFeature'] = [[3,4,5,6,7,8,9,10,11,12,13,14,15], 
#                                               [.5,.6,.7,.8,.9,1,1.1,1.2]]
# vec_methods['GetAdaptativeSystemFeature'] = [['gmm'], 
#                                                 [5,10,15,20,25,30,35,40,45]]


#%%

data_path = 'Shrec14/data/'
feat_path = "Shrec14/features/"
path_results = "results/"

#%%
for t in range(1,10): 

    with open(data_path + 'dgmsT' +'.pkl', 'rb') as f:
        dgmsT = pickle.load(f)
    with open(data_path + 'Z_train' +'.pkl', 'rb') as f:
        train_index = pickle.load(f)
    with open(data_path + 'Z_test' +'.pkl', 'rb') as f:
        test_index = pickle.load(f)
    with open(data_path + 'y_train' +'.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_path + 'y_test' +'.pkl', 'rb') as f:
        y_test = pickle.load(f)
        
    
    train_index, test_index = train_index[str(t)], test_index[str(t)]
    y_train, y_test = y_train[str(t)], y_test[str(t)]
    
    index = train_index + test_index
    diagrams_aux = dgmsT[str(t)]
    pdiagrams = dict()
    #we load the diagrams in the same format than the other datasets
    for i in index:
        pdiagrams[str(i)] = diagrams_aux[i]
    
    
    feature_dictionary = feature_computation(vec_methods, pdiagrams, "",
                                             train_index, test_index)

    with open(path_results+'SHREC14_'+str(t)+'_feature_dictionary.pkl', 'wb') as f:
      pickle.dump(feature_dictionary, f)
      
    from parameter_optimization import *
    
    complete = [
        {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
        {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
        {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
          'gamma': expon(scale=.01)},
        {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
          'degree': [2,3], 'gamma': expon(scale=.01)},
     ]
    
    onlyRF = [
        {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]}
    ]
    
    searchR =  RandomizedSearchCV(
        main_classifier(), param_distributions=complete, cv=5, n_iter=40,
        return_train_score=True, scoring='accuracy', random_state=1
    )
    

    
    best_scores=parameter_optimization(train_index, y_train, vec_methods, feature_dictionary, 
                                       searchR, normalization=True)

    
    print("Parameter optimization:",best_scores)
    
    with open(path_results+'SHREC14_best_scores.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
  
    
    n_iters = 100
    train_scores, test_scores = scores(train_index, y_train, test_index, y_test, 
                                       vec_methods, feature_dictionary, best_scores, 
                                       n_iters, normalization=True)
    
    print("The train accuracy is", train_scores)
    print("The test accuracy is", test_scores)
    
    with open(path_results+'SHREC14_'+str(t)+'_train_scores.pkl', 'wb') as f:
      pickle.dump(train_scores, f)
    with open(path_results+'SHREC14_'+str(t)+'_test_scores.pkl', 'wb') as f:
      pickle.dump(test_scores, f)

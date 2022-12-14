from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from parameter_optimization import *
from scipy.stats import uniform
from scipy.stats import expon
import pickle
from feature_computation import *
import vectorisation as vect
import pandas as pd
from direct_optimisation import main_classifier
from auxiliary_functions import *
from numpy.random import seed
     
s=1
seed(s)
    
n_iters = 100
normalization = True

#%%

data_path = 'Shrec14/data/'
feat_path = "Shrec14/features/"
path_results = "results/"

#%%

vec_parameters = dict()
vec_parameters['GetPersStats']=(),
vec_parameters['GetCarlssonCoordinatesFeature']=(),
vec_parameters['GetPersEntropyFeature'] = [[50,100,200]]
vec_parameters['GetBettiCurveFeature'] = [[50,100,200]]
vec_parameters['GetPersLifespanFeature'] = [[50,100,200]]
vec_parameters['GetAtolFeature'] = [[2,4,8,16]]
vec_parameters['GetPersTropicalCoordinatesFeature'] = [[10,50,250,500,800]]
vec_parameters['GetPersImageFeature'] = [[0.05,0.5,1],[10,20,40]] #SVM do not converge for t=9 and PersImage
vec_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
vec_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
vec_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]
vec_parameters['GetTemplateFunctionFeature'] = [[3,4,5,6,7,8,9,10,11,12,13,14,15], 
                                                [.5,.6,.7,.8,.9,1,1.1,1.2]]
vec_parameters['GetAdaptativeSystemFeature'] = [['gmm'], 
                                                [5,10,15,20,25,30,35,40,45]]

#%%
        
complete = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01)},
    {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
      'degree': [2,3], 'gamma': expon(scale=.01)},
 ]

searchR =  RandomizedSearchCV(
    main_classifier(), param_distributions=complete, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)
        
onlyForest = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100]},
 ]

searchG = GridSearchCV(
    main_classifier(), param_grid=onlyForest, cv=5,
    return_train_score=True, scoring='accuracy'
)

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
    
    func_list = [getattr(vect, keys) for keys in vec_parameters.keys()]
    for func in func_list:
        feature_dictionary = dict()
        vec_methods = dict()
        vec_methods[func.__name__] = vec_parameters[func.__name__]
        
            
        feature_dictionary = feature_computation(vec_methods, pdiagrams, "",
                                                 train_index, test_index)
    
        with open(path_results+'SHREC14_'+str(t)+'_feature_'+func.__name__+'.pkl', 'wb') as f:
          pickle.dump(feature_dictionary, f)
        
        if func==vect.GetPersImageFeature and t==9:
            best_scores=parameter_optimization(train_index, y_train, vec_methods, feature_dictionary, 
                                               searchG, normalization)
        else:
            best_scores=parameter_optimization(train_index, y_train, vec_methods, feature_dictionary, 
                                               searchR, normalization)
        
    
        
        print("Parameter optimization:",best_scores)
        
        with open(path_results+'SHREC14_'+str(t)+'_best_scores_'+func.__name__+'.pkl', 'wb') as f:
          pickle.dump(best_scores, f)
      
    
        train_scores, test_scores = scores(train_index, y_train, test_index, y_test, 
                                           vec_methods, feature_dictionary, best_scores, 
                                           n_iters, normalization)
        
        print("The train accuracy is", train_scores)
        print("The test accuracy is", test_scores)
        
        with open(path_results+'SHREC14_'+str(t)+'_train_scores_'+func.__name__+'.pkl', 'wb') as f:
          pickle.dump(train_scores, f)
        with open(path_results+'SHREC14_'+str(t)+'_test_scores_'+func.__name__+'.pkl', 'wb') as f:
          pickle.dump(test_scores, f)

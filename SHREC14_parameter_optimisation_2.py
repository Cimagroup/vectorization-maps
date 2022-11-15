import pickle
import pandas as pd
from vectorisation import *
import numpy as np
from numpy.random import seed
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from direct_optimisation import main_classifier
from SHREC14_tropical_optimisation import tropical_classifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform
from scipy.stats import expon
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

seed(1)
data_path = 'Shrec14/data/'
feat_path = "Shrec14/features/"

#%%
#Load the labels
with open(feat_path + 'Z_train' +'.pkl', 'rb') as f:
    Z_train = pickle.load(f)
with open(feat_path + 'y_train' +'.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(feat_path + 'dgmsT' +'.pkl', 'rb') as f:
    dgmsT = pickle.load(f)  

#%%
#Different possible grids (some methods do not converge for some 
#hyperparameters)
onlyRF = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]}
    ]

complete = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01)},
    {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
      'degree': [2,3], 'gamma': expon(scale=.01)},
 ]

tropical_params = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(0,200)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01), 'r': uniform(0,200)}
]

searchR = RandomizedSearchCV(
    main_classifier(), param_distributions=complete, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)

searchG = GridSearchCV(
    main_classifier(), param_grid=onlyRF, cv=5,
    return_train_score=True, scoring='accuracy'
)

searchT = lambda x : RandomizedSearchCV(
    tropical_classifier(dgmsT=dgmsT, t=x), param_distributions=tropical_params, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetPersImageFeature'] = [10,25,50,100,200]
hyper_parameters['GetAtolFeature'] = [2,4,8,16]
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]
hyper_parameters['GetTemplateFunctionFeature'] = [[3,4,5,6,7,8,9,10,11,12,13,14,15], 
                                              [.5,.6,.7,.8,.9,1,1.1,1.2]]
hyper_parameters['GetAdaptativeSystemFeature'] = [['gmm'], 
                                                [1,2,3,4, 5,10,15,20,25,30,35,40,45,50]]

methods_no_param = [
                    # GetPersStats,
                    # GetCarlssonCoordinatesFeature
                    ]

methods_one_param = [
                     # GetPersEntropyFeature,
                     # GetBettiCurveFeature,
                     # GetPersLifespanFeature,
                     # GetTopologicalVectorFeature,
                     # GetPersImageFeature,
                     # GetAtolFeature
                     ]

methods_two_param = [
                     # GetPersSilhouetteFeature,
                     # GetComplexPolynomialFeature,
                     # GetPersLandscapeFeature,
                     # GetTemplateFunctionFeature,
                     # GetAdaptativeSystemFeature
                     ]

#%%

def shrec14_optimization(func, search, Z_train, y_train, 
                         hps, t):
    if func==GetPersTropicalCoordinatesFeature:
        X_train=Z_train
        search.fit(X_train, y_train)
        best_scores = (search.best_params_, search.best_score_)
    else:        
        with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
            features = pickle.load(f)
        
        if func in methods_no_param:
            X_train = []
            for i in Z_train:
                X_train.append(features[str(t)+'_'+str(i)])
                
            mm_scaler = MinMaxScaler()
            X_train = mm_scaler.fit_transform(X_train)
            search.fit(X_train, y_train)
            best_scores = (search.best_params_, search.best_score_)
    
        if func in methods_one_param:
            best_scores = {}
            for p in hps[func.__name__]:
                X_train = []
                for i in Z_train:
                    X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])

                mm_scaler = MinMaxScaler()
                X_train = mm_scaler.fit_transform(X_train)
                search.fit(X_train, y_train)
                best_scores[str(p)] = (search.best_params_, search.best_score_)
                
        if func in methods_two_param:
            best_scores = {}
            for p in hps[func.__name__][0]:
                if (func==GetAdaptativeSystemFeature) and (p=='hdb'):
                    #for that p, the Adaptative System do not require a q value,
                    #so we use the dummy value q=25
                    q=25
                    X_train = []
                    for i in Z_train:
                        X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
                    mm_scaler = MinMaxScaler()
                    X_train = mm_scaler.fit_transform(X_train)
                    search.fit(X_train, y_train)
                    best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
                else:    
                    for q in hps[func.__name__][1]:
                        X_train = []
                        for i in Z_train:
                            X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
                        mm_scaler = MinMaxScaler()
                        X_train = mm_scaler.fit_transform(X_train)
                        search.fit(X_train, y_train)
                        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
  
    return best_scores


#%%

func_list = methods_no_param + methods_one_param + methods_two_param
#func_list = []
for func in func_list:
    print(func.__name__)
    best_scores = {} 
    for t in range(1,11):
        if (func != GetAtolFeature) or (func != GetTemplateFunctionFeature) or (t<10):
        #Atol and Template Functions could not be computed when t=10
            print(t)
            best_scores[str(t)]=shrec14_optimization(func=func, search=searchR, 
                                                     Z_train=Z_train[str(t)], 
                                                     y_train=y_train[str(t)],
                                                     hps=hyper_parameters, t=t)
        
    print(best_scores)
    print()
    with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
#%%

func = GetPersImageFeature
print(func.__name__)
best_scores = {}
for t in range(1,11):
    print(t)
    best_scores[str(t)]=shrec14_optimization(func=func, search=searchG, 
                                             Z_train=Z_train[str(t)], 
                                             y_train=y_train[str(t)],
                                             hps=hyper_parameters, t=t)
    
print(best_scores)
print()
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%  
func = GetPersTropicalCoordinatesFeature
print(func.__name__)
best_scores = {}
for t in range(1,11):
    print(t)
    best_scores[str(t)] = shrec14_optimization(func=func, search=searchT(t),
                                     Z_train=Z_train[str(t)], 
                                     y_train=y_train[str(t)],
                                     hps=hyper_parameters, t=t)

print(best_scores)
print()

with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
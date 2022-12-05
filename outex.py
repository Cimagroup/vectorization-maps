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
from numpy.random import choice, seed
from copy import deepcopy

number_labels = 10
# number_labels = 68
s=1
seed(s)

#%%

vec_methods = dict()
# vec_methods['GetPersStats']=(),
# vec_methods['GetCarlssonCoordinatesFeature']=(),
# vec_methods['GetPersEntropyFeature'] = [[50,100,200]]
# vec_methods['GetBettiCurveFeature'] = [[50,100,200]]
# vec_methods['GetPersLifespanFeature'] = [[50,100,200]]
# vec_methods['GetTopologicalVectorFeature'] = [[5, 10, 20]]
vec_methods['GetAtolFeature'] = [[16]]#[[2,4,8,16,32,64]]
# vec_methods['GetPersTropicalCoordinatesFeature'] = [[10,50,250,500,800]]
# vec_methods['GetPersImageFeature'] = [[0.001,0.01,0.2,1],[20,50,70]]
# vec_methods['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
# vec_methods['GetComplexPolynomialFeature'] = [[5,10,20],['R', 'S', 'T']]
# vec_methods['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]
# vec_methods['GetTemplateFunctionFeature'] = [[50],[25]]#[[35,50,65], [20,25,30]]
# vec_methods['GetAdaptativeSystemFeature'] = [['gmm'], [20]]# [10,20,30,40,50]]


#%%

path_feat = 'Outex-TC-00024/features/'
path_data = "Outex-TC-00024/data/000/"
path_diag = "Outex-TC-00024/pdiagrams/"
path_results = "results/"

#%%
# Pick the 68 labels or 10 random depending on number_labels.

labels = range(68)
if number_labels==10:
    labels = choice(labels, size=(10), replace = False)

train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                    usecols=[1]).to_numpy().flatten().tolist())
test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                    usecols=[1]).to_numpy().flatten().tolist())
if number_labels==10:
    train_index = np.array([i for i in range(len(train_labels)) if train_labels[i] in labels])
    test_index = np.array([i for i in range(len(test_labels)) if test_labels[i] in labels])
    
label_list = np.hstack([train_labels[train_index], test_labels[test_index]])

train_index, test_index, y_train, y_test = train_test_split(range(len(label_list)), 
                                                    label_list, test_size=0.3, 
                                                    random_state=0)
index = train_index + test_index

#%%
pdiagrams = dict()

for i in index:
    pdiagrams["pdiag_l_d0_"+str(i)]= safe_load(path_diag + "l_d0_"+str(i))
    pdiagrams["pdiag_l_d1_"+str(i)]= safe_load(path_diag + "l_d1_"+str(i))
    pdiagrams["pdiag_u_d0_"+str(i)]= safe_load(path_diag + "u_d0_"+str(i))
    pdiagrams["pdiag_u_d1_"+str(i)]= safe_load(path_diag + "u_d1_"+str(i))
    
#%%

feature_dictionary_l_d0 = feature_computation(vec_methods, pdiagrams, "pdiag_l_d0_",
                                              train_index, test_index)
feature_dictionary_l_d1 = feature_computation(vec_methods, pdiagrams, "pdiag_l_d1_",
                                              train_index, test_index)
feature_dictionary_u_d0 = feature_computation(vec_methods, pdiagrams, "pdiag_u_d0_",
                                              train_index, test_index)
feature_dictionary_u_d1 = feature_computation(vec_methods, pdiagrams, "pdiag_u_d1_",
                                              train_index, test_index)

#%%
func_list = [getattr(vect, keys) for keys in vec_methods.keys()]
feature_dictionary = dict()
for func in func_list:
    features= dict()
    func_parameters = load_parameters(func,vec_methods)
    for p in func_parameters:
        for i in index:
            features[str(i)] = np.hstack(
                                    [
                                        feature_dictionary_l_d0[func.__name__+'_'+str(p)][str(i)],
                                        feature_dictionary_l_d1[func.__name__+'_'+str(p)][str(i)],
                                        feature_dictionary_u_d0[func.__name__+'_'+str(p)][str(i)],
                                        feature_dictionary_u_d1[func.__name__+'_'+str(p)][str(i)]
                                    ]
                                    )
        
        feature_dictionary[func.__name__+'_'+str(p)]= deepcopy(features)
        
    with open(path_results+'OUTEX'+str(number_labels)+'_feature_dictionary.pkl', 'wb') as f:
      pickle.dump(feature_dictionary, f)
      
#%%

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

searchG = GridSearchCV(
    main_classifier(), param_grid=onlyRF, cv=5,
    return_train_score=True, scoring='accuracy'
)


#%%

best_scores=parameter_optimization(train_index, y_train, vec_methods, feature_dictionary, 
                                   searchR, normalization=True)


#%%

print("Parameter optimization:",best_scores)

with open(path_results+'OUTEX'+str(number_labels)+'_best_scores.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
#%%

n_iters = 100
train_scores, test_scores = scores(train_index, y_train, test_index, y_test, 
                                   vec_methods, feature_dictionary, best_scores, 
                                   n_iters, normalization=True)

#%%

print("The train accuracy is", train_scores)
print("The test accuracy is", test_scores)

with open(path_results+'OUTEX'+str(number_labels)+'_train_scores.pkl', 'wb') as f:
  pickle.dump(train_scores, f)
with open(path_results+'OUTEX'+str(number_labels)+'_test_scores.pkl', 'wb') as f:
  pickle.dump(test_scores, f)
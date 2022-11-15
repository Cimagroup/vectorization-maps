from direct_optimisation import main_classifier
from OUTEX_tropical_optimisation import tropical_classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from scipy.stats import expon
from vectorisation import * 
import numpy as np
import pandas as pd
import pickle
import copy
from numpy.random import choice, seed

path_feat = 'Outex-TC-00024/features/'
path_data = "Outex-TC-00024/data/000/"
    
seed(1)  

#%%
#Different possible grids (some methods do not converge for some 
#hyperparameters)

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

tropical_params = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01), 'r': uniform(1,1000)},
]

searchR = RandomizedSearchCV(
    main_classifier(), param_distributions=complete, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)

searchG = GridSearchCV(
    main_classifier(), param_grid=onlyRF, cv=5,
    return_train_score=True, scoring='accuracy'
)

searchT = RandomizedSearchCV(
    tropical_classifier(), param_distributions=tropical_params, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetAtolFeature'] = [2,4,8,16,32,64]
hyper_parameters['GetPersImageFeature'] = [25,50,100,150,200]
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]
hyper_parameters['GetTemplateFunctionFeature'] = [[35,50,65], [20,25,30]]
hyper_parameters['GetAdaptativeSystemFeature'] = [['gmm'],#, 'hdb'], 
                                                [1,2,3,4, 5,10,15,20,25,30,35,40,45,50]]


#%%

methods_no_param = [
                    GetPersStats,
                    GetCarlssonCoordinatesFeature
                    ]

methods_one_param = [
                       GetPersEntropyFeature,
                       GetBettiCurveFeature,
                       GetPersLifespanFeature,
                       GetTopologicalVectorFeature,
                       GetPersImageFeature,
                       GetAtolFeature
                     ]

methods_two_param = [
                       GetPersSilhouetteFeature,
                       GetComplexPolynomialFeature,
                        GetPersLandscapeFeature,
                        GetTemplateFunctionFeature,
                        GetAdaptativeSystemFeature
                     ]


#%%

def outex_optimization(func, search, Z_train, y_train, hps,n):
    
    if func==GetPersTropicalCoordinatesFeature:
        X_train=Z_train
        search.fit(X_train, y_train)
        best_scores = (search.best_params_, search.best_score_)
    else:
        if (func==GetAdaptativeSystemFeature)or(func==GetTemplateFunctionFeature):
            with open(path_feat + func.__name__ +str(n)+'_l_d0.pkl', 'rb') as f:
                features_l_d0 = pickle.load(f)
            with open(path_feat + func.__name__ +str(n)+'_l_d1.pkl', 'rb') as f:
                features_l_d1 = pickle.load(f)
            with open(path_feat + func.__name__ +str(n)+'_u_d0.pkl', 'rb') as f:
                features_u_d0 = pickle.load(f)
            with open(path_feat + func.__name__ +str(n)+'_u_d1.pkl', 'rb') as f:
                features_u_d1 = pickle.load(f)
        else:        
            with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
                features_l_d0 = pickle.load(f)
            with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
                features_l_d1 = pickle.load(f)
            with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
                features_u_d0 = pickle.load(f)
            with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
                features_u_d1 = pickle.load(f)
        
        if func in methods_no_param:
            X_train = []
            for i in Z_train:
                X_train.append(
                    np.hstack(
                        [
                            features_l_d0[str(i)],
                            features_l_d1[str(i)],
                            features_u_d0[str(i)],
                            features_u_d1[str(i)]
                        ]
                        ))
                
            mm_scaler = MinMaxScaler()
            X_train = mm_scaler.fit_transform(X_train)
            search.fit(X_train, y_train)
            best_scores = (search.best_params_, search.best_score_)
    
        if func in methods_one_param:
            best_scores = {}
            for p in hps[func.__name__]:
                X_train = []
                for i in Z_train:
                    X_train.append(
                        np.hstack(
                            [
                                features_l_d0[str(i)+'_'+str(p)],
                                features_l_d1[str(i)+'_'+str(p)],
                                features_u_d0[str(i)+'_'+str(p)],
                                features_u_d1[str(i)+'_'+str(p)]
                            ]
                            ))
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
                        X_train.append(
                            np.hstack(
                                [
                                    features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                                    features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                                    features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                                    features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                                ]
                                ))
                    mm_scaler = MinMaxScaler()
                    X_train = mm_scaler.fit_transform(X_train)
                    search.fit(X_train, y_train)
                    best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
                else:    
                    for q in hps[func.__name__][1]:
                        X_train = []
                        for i in Z_train:
                            X_train.append(
                                np.hstack(
                                    [
                                        features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                                        features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                                        features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                                        features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                                    ]
                                    ))
                        mm_scaler = MinMaxScaler()
                        X_train = mm_scaler.fit_transform(X_train)
                        search.fit(X_train, y_train)
                        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
  
    return best_scores

#%%
# OUTEX10 labels, train and test set

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
                                                    random_state=0)

#%%
# OUTEX10 calculations

func_list = methods_no_param + methods_one_param + methods_two_param
#func_list = []

for func in func_list:
    print(func.__name__)
    best_scores = outex_optimization(func=func, search=searchR, 
                                     Z_train=Z_train, y_train=y_train,
                                     hps=hyper_parameters, n=10)
    print(best_scores)
    print()
    
    with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)

#%%
func = GetPersImageFeature
print(func.__name__)
best_scores = outex_optimization(func=func, search=searchG,
                                  Z_train=Z_train, y_train=y_train,
                                  hps=hyper_parameters, n=10)

print(best_scores)
print()

with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f) 

#%%      

func = GetPersTropicalCoordinatesFeature
best_scores = outex_optimization(func=func, search=searchT,
                                  Z_train=Z_train, y_train=y_train,
                                  hps=hyper_parameters, n=10)
print(func.__name__)
print(best_scores)
print()

# with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
#   pickle.dump(best_scores, f)

#%%
# OUTEX68 labels, test and train

train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                    usecols=[1]).to_numpy().flatten().tolist())
test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                    usecols=[1]).to_numpy().flatten().tolist())
labels = np.hstack([train_labels, test_labels])
Z_train, Z_test, y_train, y_test = train_test_split(range(2720), labels, 
                                                    test_size=0.3, 
                                                    random_state=0)

#%%
# OUTEX68 calculations

func_list = methods_no_param + methods_one_param + methods_two_param
#func_list = []

for func in func_list:
    print(func.__name__)
    best_scores = outex_optimization(func=func, search=searchR,
                                     Z_train=Z_train, y_train=y_train,
                                     hps=hyper_parameters, n=68)
    
    print(best_scores)
    print()
    
    with open(path_feat + func.__name__ + '_68_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
    
#%%

func = GetPersImageFeature
print(func.__name__)
best_scores = outex_optimization(func=func, search=searchG,
                                  Z_train=Z_train, y_train=y_train,
                                  hps=hyper_parameters, n=68)

print(best_scores)
print()

with open(path_feat + func.__name__ + '_68_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)      

#%%      
func = GetPersTropicalCoordinatesFeature
print(func.__name__)
best_scores = outex_optimization(func=func, search=searchT,
                                  Z_train=Z_train, y_train=y_train,
                                  hps=hyper_parameters, n=68)

print(best_scores)
print()

with open(path_feat + func.__name__ + '_68_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

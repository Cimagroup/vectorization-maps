from direct_optimisation import main_classifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from scipy.stats import expon
from vectorisation import * 
import numpy as np
import pandas as pd
import pickle
from numpy.random import choice, seed

path_feat = 'Outex-TC-00024/features/'
path_data = "Outex-TC-00024/data/000/"

seed(1)
labels = range(68)
labels = choice(labels, size=(10), replace = False)

#%%
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

onlyForest = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
 ]

noPoly = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01)}
 ]

noL =  [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01)},
    {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
      'degree': [2,3], 'gamma': expon(scale=.01)},
 ]

forestRBF = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01)}
 ]

forestL = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)}
 ]

searchR = lambda pg : RandomizedSearchCV(
    main_classifier(), param_distributions=pg, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)

searchG = lambda pg :  GridSearchCV(
    main_classifier(), param_grid=pg, cv=5,
    return_train_score=True, scoring='accuracy'
)

#%%
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
hyper_parameters['GetTentFunctionFeature'] = [[20,35,50,65,80], [5,7,9]]
hyper_parameters['GetTemplateSystemFeature'] = [['gmm', 'hdb'], 
                                                [1,2,3,4, 5,10,15,20,25,30,35,40,45,50]]

#%%

func = GetPersStats

print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)

search =  searchR(complete)

best_scores = {}
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
    
X_test = []
for i in Z_test:
    X_test.append(
        np.hstack(
            [
                features_l_d0[str(i)],
                features_l_d1[str(i)],
                features_u_d0[str(i)],
                features_u_d1[str(i)]
            ]
            ))    


search.fit(X_train, y_train)

best_scores = (search.best_params_, search.best_score_)
print(best_scores)
    

with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
#Methods with no parameter
func = GetCarlssonCoordinatesFeature
print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    

search =  searchG(onlyForest)

best_scores = {}
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
    
X_test = []
for i in Z_test:
    X_test.append(
        np.hstack(
            [
                features_l_d0[str(i)],
                features_l_d1[str(i)],
                features_u_d0[str(i)],
                features_u_d1[str(i)]
            ]
            ))    


search.fit(X_train, y_train)

best_scores = (search.best_params_, search.best_score_)
print(best_scores)
    

with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%

func = GetPersEntropyFeature
print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)

search =  searchR(complete)

best_scores = {}
for p in hyper_parameters[func.__name__]:
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
        
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
                    features_l_d0[str(i)+'_'+str(p)],
                    features_l_d1[str(i)+'_'+str(p)],
                    features_u_d0[str(i)+'_'+str(p)],
                    features_u_d1[str(i)+'_'+str(p)]
                ]
                ))    


    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func = GetBettiCurveFeature

print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    
search = searchR(complete)

best_scores = {}
for p in hyper_parameters[func.__name__]:
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
        
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
                    features_l_d0[str(i)+'_'+str(p)],
                    features_l_d1[str(i)+'_'+str(p)],
                    features_u_d0[str(i)+'_'+str(p)],
                    features_u_d1[str(i)+'_'+str(p)]
                ]
                ))    


    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
#%%
func = GetTopologicalVectorFeature
print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    
search = searchG(onlyForest)

best_scores = {}
for p in hyper_parameters[func.__name__]:
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
        
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
                    features_l_d0[str(i)+'_'+str(p)],
                    features_l_d1[str(i)+'_'+str(p)],
                    features_u_d0[str(i)+'_'+str(p)],
                    features_u_d1[str(i)+'_'+str(p)]
                ]
                ))    


    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)           

#%%
func = GetPersLifespanFeature
print(func.__name__)
    
with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)


search =  searchR(complete)
best_scores = {}
for p in hyper_parameters[func.__name__]:
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
        
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
                    features_l_d0[str(i)+'_'+str(p)],
                    features_l_d1[str(i)+'_'+str(p)],
                    features_u_d0[str(i)+'_'+str(p)],
                    features_u_d1[str(i)+'_'+str(p)]
                ]
                ))    


    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
      
#%%
func = GetAtolFeature

print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)

best_scores = {}
for p in hyper_parameters[func.__name__]:
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
        
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
                    features_l_d0[str(i)+'_'+str(p)],
                    features_l_d1[str(i)+'_'+str(p)],
                    features_u_d0[str(i)+'_'+str(p)],
                    features_u_d1[str(i)+'_'+str(p)]
                ]
                ))   
    if p<=4:
        search = searchR(forestRBF)
    else:
        search = searchR(noPoly)
    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%

func = GetPersImageFeature
print(func.__name__)
    
with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)


search = searchR(complete)
best_scores = {}
for p in hyper_parameters[func.__name__]:
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
        
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
                    features_l_d0[str(i)+'_'+str(p)],
                    features_l_d1[str(i)+'_'+str(p)],
                    features_u_d0[str(i)+'_'+str(p)],
                    features_u_d1[str(i)+'_'+str(p)]
                ]
                ))    

    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func = GetPersSilhouetteFeature
print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    
search = searchR(complete)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
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
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))    

        X_train = float64to32(X_train)
        X_test = float64to32(X_test)
        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
      
#%%
func = GetComplexPolynomialFeature
print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    
search = searchG(onlyForest)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
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
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))    

        X_train = float64to32(X_train)
        X_test = float64to32(X_test)
        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func = GetPersLandscapeFeature

print(func.__name__)

with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    
search = searchR(complete)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
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
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))    

        X_train = float64to32(X_train)
        X_test = float64to32(X_test)
        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func = GetTentFunctionFeature

print(func.__name__)

with open(path_feat + func.__name__ +'10_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '10_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '10_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '10_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    

search = searchR(complete)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
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
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))    

        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + 'hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%

func = GetTemplateSystemFeature

print(func.__name__)

with open(path_feat + func.__name__ +'10_l_d0.pkl', 'rb') as f:
    features_l_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '10_l_d1.pkl', 'rb') as f:
    features_l_d1 = pickle.load(f)
with open(path_feat + func.__name__ + '10_u_d0.pkl', 'rb') as f:
    features_u_d0 = pickle.load(f)
with open(path_feat + func.__name__ + '10_u_d1.pkl', 'rb') as f:
    features_u_d1 = pickle.load(f)
    



best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    if p=='gmm':
        for q in hyper_parameters[func.__name__][1]:
            if q<20:
                search = searchR(forestRBF)
            else:
                search = searchR(complete)
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
                
            X_test = []
            for i in Z_test:
                X_test.append(
                    np.hstack(
                        [
                            features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
                            features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
                            features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
                            features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
                        ]
                        ))    
    
            search.fit(X_train, y_train)
            best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
            print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    # else:
    #     q = 25
    #     search = searchR(complete)
    #     X_train = []
    #     for i in Z_train:
    #         X_train.append(
    #             np.hstack(
    #                 [
    #                     features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
    #                     features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
    #                     features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
    #                     features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
    #                 ]
    #                 ))
            
    #     X_test = []
    #     for i in Z_test:
    #         X_test.append(
    #             np.hstack(
    #                 [
    #                     features_l_d0[str(i)+'_'+str(p)+'_'+str(q)],
    #                     features_l_d1[str(i)+'_'+str(p)+'_'+str(q)],
    #                     features_u_d0[str(i)+'_'+str(p)+'_'+str(q)],
    #                     features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]
    #                 ]
    #                 ))    

        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + 'hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)  
  
#%%
func = GetPersTropicalCoordinatesFeature
from OUTEX_tropical_optimisation import tropical_classifier

parameters = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000), 'r': uniform(1,1000)}
 ]

X_train, X_test = Z_train, Z_test
search = RandomizedSearchCV(
    tropical_classifier(), param_distributions = parameters, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy', random_state=1
)

search.fit(X_train, y_train)

best_scores = (search.best_params_, search.best_score_)
print(best_scores)
    

with open(path_feat + func.__name__ + '_10_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
# GetPersStats
# 'base_estimator': 'RF', 'n_estimators': 100}, 0.9821428571428571

# GetCarlssonCoordinatesFeature
# {'base_estimator': 'RF', 'n_estimators': 200}, 0.9428571428571428

# GetPersEntropyFeature
# 50  : ({'C': 936.5390708060319, 'base_estimator': 'SVM', 'gamma': 0.01872823656893796, 'kernel': 'rbf'}, 0.9535714285714285)
# 100  : ({'C': 936.5390708060319, 'base_estimator': 'SVM', 'gamma': 0.01872823656893796, 'kernel': 'rbf'}, 0.9428571428571427)
# 200  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.9464285714285714)

# GetBettiCurveFeature
# 50  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9392857142857143)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9321428571428573)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9321428571428572)

# GetTopologicalVectorFeature
# 5  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7678571428571429)
# 10  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7821428571428571)
# 20  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8142857142857143)

# GetPersLifespanFeature
# 50  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.95)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9464285714285714)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9428571428571428)

# GetAtolFeature
# 2  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9178571428571429)
# 4  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9321428571428572)
# 8  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9607142857142857)
# 16  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9535714285714285)
# 32  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9678571428571427)
# 64  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9821428571428571)

# GetPersImageFeature
# 25  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9178571428571429)
# 50  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9107142857142858)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9107142857142858)
# 150  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9071428571428573)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9107142857142858)

# GetPersSilhouetteFeature
# 50_0  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9678571428571429)
# 50_1  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.9678571428571429)
# 50_2  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9392857142857143)
# 50_5  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8964285714285716)
# 50_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9)
# 50_20  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8535714285714284)
# 100_0  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9571428571428571)
# 100_1  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.9678571428571429)
# 100_2  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9357142857142857)
# 100_5  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8964285714285716)
# 100_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8892857142857142)
# 100_20  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8642857142857142)
# 200_0  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9464285714285714)
# 200_1  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.9678571428571429)
# 200_2  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9357142857142857)
# 200_5  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8964285714285716)
# 200_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8678571428571427)
# 200_20  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8642857142857142)

# GetComplexPolynomialFeature
# 5_R  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9571428571428571)
# 5_S  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8928571428571429)
# 5_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.3857142857142858)
# 10_R  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9571428571428571)
# 10_S  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9035714285714287)
# 10_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.39285714285714285)
# 20_R  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9392857142857143)
# 20_S  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9)
# 20_T  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.44642857142857145)

# GetPersLandscapeFeature
# 50_2  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9178571428571429)
# 50_5  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9285714285714286)
# 50_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9285714285714286)
# 50_20  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9321428571428572)
# 100_2  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9178571428571429)
# 100_5  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9357142857142857)
# 100_10  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.9214285714285715)
# 100_20  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9321428571428572)
# 200_2  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9178571428571429)
# 200_5  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.925)
# 200_10  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9250000000000002)
# 200_20  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9321428571428572)
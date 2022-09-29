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

path_feat = 'Outex-TC-00024/features/'
path_data = "Outex-TC-00024/data/000/"

#%%
train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                    usecols=[1]).to_numpy().flatten().tolist())
test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                   usecols=[1]).to_numpy().flatten().tolist())
labels = np.hstack([train_labels, test_labels])
Z_train, Z_test, y_train, y_test = train_test_split(range(2720), labels, 
                                                    test_size=0.3, 
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
    main_classifier(), param_distributions=pg, cv=5, n_iter=5,
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
hyper_parameters['GetAtolFeature'] = [2,4,8,16]
hyper_parameters['GetPersImageFeature'] = [50,100,150,200,250]
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]

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
    

with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
    

with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
    if p==4:
        search = searchR(noL)
    else:
        search = searchR(complete)
    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
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
    

with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
# GetPersStats
# ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8781461527835337)

# GetCarlssonCoordinatesFeature
# ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8319284431551319)

#GetPersEntropyFeature
# '50': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8266832435419257), 
# '100': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.828258046691532), 
# '200': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8256292305567067)

#GetBettiCurveFeature
# 50  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.7888713910761155)
# 100  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.7988437629506837)
# 200  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7962121840033154)

#GetPersLifeSpan
# 50  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8293106782704793)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8345614035087718)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8303619284431552)

#GetPersImage
# 100  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7862384307224756)
# 150  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7888589584196712)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7941124464705069)
# 250  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7956941566514711)

#GetAtolFeature
# 2  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.5761431136897361)
# 4  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.4684804531012571)
# 8  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.39914905373670395)
# 16  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.33873877607404335)

#GetPersSilhouette
# 50_0  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8214256112722751)
# 50_1  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.836130681033292)
# 50_2  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7998991573421743)
# 50_5  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7263682829120045)
# 50_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.670685177510706)
# 50_20  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.664391490537367)
# 100_0  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8182773863793342)
# 100_1  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.8366583782290371)
# 100_2  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8019961320624395)
# 100_5  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7310940737670949)
# 100_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.682247548003868)
# 100_20  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.6664926094764471)
# 200_0  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8193258737394669)
# 200_1  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8345627849150435)
# 200_2  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8025210664456417)
# 200_5  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.727416770272137)
# 200_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.6754137311783396)
# 200_20  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.6680632684072385)

#GetComplexPolynomial
# 5_R  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7788852051388313)
# 5_S  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7216328222130128)
# 5_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.23529216742644016)
# 10_R  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7720582953446609)
# 10_S  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7384486807570108)
# 10_T  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.23530321867661277)
# 20_R  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7263669015057329)
# 20_S  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7605014504765851)
# 20_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.26944467467882305)

#Landscape
# 50_2  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.7815209283050144)
# 50_5  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.806199751346871)
# 50_10  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8214352811161764)
# 50_20  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8308882442326289)
# 100_2  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.7825721784776902)
# 100_5  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8046235667909933)
# 100_10  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8214311368973615)
# 100_20  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8287871252935488)
# 200_2  : ({'C': 1000.0405153241447, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009688387165373345, 'kernel': 'poly'}, 0.7809987567343556)
# 200_5  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8056734355573975)
# 200_10  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8209048211078878)
# 200_20  : ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8287871252935488)

#Tropical
# ({'base_estimator': 'RF', 'n_estimators': 200, 'r': 492.5731592803383}, 0.8639632545931759)
      


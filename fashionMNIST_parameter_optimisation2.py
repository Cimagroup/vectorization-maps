from direct_optimisation import main_classifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from scipy.stats import expon
import pickle
import numpy as np
from fashion_mnist import mnist_reader
from vectorisation import *
import pandas as pd

path_feat = "fashion_mnist/features/"
path_diag= "fashion_mnist/pdiagrams/"


#%%
_, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
_, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

n_train = len(y_train)
n_total = len(y_train) + len(y_test)

Z_train = range(n_train)
Z_test = range(n_train, n_total)

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
hyper_parameters['GetPersEntropyFeature'] = [15,30,50]
hyper_parameters['GetBettiCurveFeature'] = [15,30,50]
hyper_parameters['GetPersLifespanFeature'] = [15,30,50]
hyper_parameters['GetTopologicalVectorFeature'] = [3, 5, 10]
hyper_parameters['GetAtolFeature'] = [2,4,8,16]
hyper_parameters['GetPersImageFeature'] = [3,6,12,20]
hyper_parameters['GetPersSilhouetteFeature'] = [[15,30,50], [0,1,2,5]]
hyper_parameters['GetComplexPolynomialFeature'] = [[3, 5, 10],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[15,30,50], [1,2,3,5]]
hyper_parameters['GetTemplateFunctionFeature'] = [[2,3,5], [.5, 1, 2]]
hyper_parameters['GetAdaptativeSystemFeature'] = [['gmm'],#, 'hdb'], 
                                                [1,2,3,4,5,10,15]]


method = onlyForest

#%%


func = GetPersStats

print(func.__name__)

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)

search =  searchG(method)

best_scores = {}
X_train = []
for i in Z_train:
    X_train.append(
        np.hstack(
            [
#                features_l[str(i)],
                features_u[str(i)]
            ]
            ))
X_test = []
for i in Z_test:
    X_test.append(
        np.hstack(
            [
#                features_l[str(i)],
                features_u[str(i)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    

search =  searchG(method)

best_scores = {}
X_train = []
for i in Z_train:
    X_train.append(
        np.hstack(
            [
#                features_l[str(i)],
                features_u[str(i)]
            ]
            ))

X_test = []
for i in Z_test:
    X_test.append(
        np.hstack(
            [
#                features_l[str(i)],
                features_u[str(i)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)

search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__]:
    X_train = []
    for i in Z_train:
        X_train.append(
            np.hstack(
                [
                #    features_l[str(i)+'_'+str(p)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
    #                features_l[str(i)],
                    features_u[str(i)+'_'+str(p)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    
search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__]:
    X_train = []
    for i in Z_train:
        X_train.append(
            np.hstack(
                [
                #    features_l[str(i)+'_'+str(p)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))    
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
    #                features_l[str(i)],
                    features_u[str(i)+'_'+str(p)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    
search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__]:
    X_train = []
    for i in Z_train:
        X_train.append(
            np.hstack(
                [
#                    features_l[str(i)+'_'+str(p)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
    #                features_l[str(i)],
                    features_u[str(i)+'_'+str(p)]
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
    
#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)


search =  searchG(method)
best_scores = {}
for p in hyper_parameters[func.__name__]:
    X_train = []
    for i in Z_train:
        X_train.append(
            np.hstack(
                [
#                    features_l[str(i)+'_'+str(p)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))

    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
    #                features_l[str(i)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))


    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)

#%%
func = GetAtolFeature

print(func.__name__)

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)

best_scores = {}
for p in hyper_parameters[func.__name__]:
    X_train = []
    for i in Z_train:
        X_train.append(
            np.hstack(
                [
#                    features_l[str(i)+'_'+str(p)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
    #                features_l[str(i)],
                    features_u[str(i)+'_'+str(p)]
                ]
                ))

    
    if p<=4:
        search =  searchG(method)
    else:
        search =  searchG(method)
    search.fit(X_train, y_train)

    best_scores[str(p)] = (search.best_params_, search.best_score_)
    print(str(p), ' :', best_scores[str(p)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
  
#%%

func = GetPersImageFeature
print(func.__name__)
    
#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)


search =  searchG(method)
best_scores = {}
for p in hyper_parameters[func.__name__]:
    X_train = []
    for i in Z_train:
        X_train.append(
            np.hstack(
                [
#                    features_l[str(i)+'_'+str(p)],
                    features_u[str(i)+'_'+str(p)]
                ]
                )) 
    X_test = []
    for i in Z_test:
        X_test.append(
            np.hstack(
                [
    #                features_l[str(i)],
                    features_u[str(i)+'_'+str(p)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    
search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
#                        features_l[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
    #                features_l[str(i)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    
search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
#                        features_l[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))

        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
    #                features_l[str(i)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
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

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    
search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
#                        features_l[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
    #                features_l[str(i)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                ))
            
        X_train = float64to32(X_train)
        X_test = float64to32(X_test)
        np.savetxt("FMNIST_"+func.__name__+"_"+str(p)+"_"+str(q)+"_X_train.txt",X_train)
        np.savetxt("FMNIST_"+func.__name__+"_"+str(p)+"_"+str(q)+"_X_test.txt",X_test)
        
        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
   
#%%

func = GetTemplateFunctionFeature

print(func.__name__)

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    

search =  searchG(method)

best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
#                        features_l[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
    #                features_l[str(i)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                ))
            
        
        
        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
  
#%%

func = GetAdaptativeSystemFeature

print(func.__name__)

#with open(path_feat + func.__name__ +'_l.pkl', 'rb') as f:
#    features_l = pickle.load(f)
with open(path_feat + func.__name__ + '_u.pkl', 'rb') as f:
    features_u = pickle.load(f)
    



best_scores = {}
for p in hyper_parameters[func.__name__][0]:
    if p=='gmm':
        for q in hyper_parameters[func.__name__][1]:
            if q<20:
                search =  searchG(method)
            else:
                search =  searchG(method)
            X_train = []
            for i in Z_train:
                X_train.append(
                    np.hstack(
                        [
#                            features_l[str(i)+'_'+str(p)+'_'+str(q)],
                            features_u[str(i)+'_'+str(p)+'_'+str(q)]
                        ]
                        ))  
            
            X_test = []
            for i in Z_test:
                X_test.append(
                    np.hstack(
                        [
        #                features_l[str(i)],
                            features_u[str(i)+'_'+str(p)+'_'+str(q)]
                        ]
                    ))
            
           
            
            search.fit(X_train, y_train)
            best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
            print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    else:
        q = 25
        search =  searchG(method)
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
#                        features_l[str(i)+'_'+str(p)+'_'+str(q)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                    ))
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
    #                features_l[str(i)],
                        features_u[str(i)+'_'+str(p)+'_'+str(q)]
                    ]
                ))
        
        
        search.fit(X_train, y_train)
        best_scores[str(p)+'_'+str(q)] = (search.best_params_, search.best_score_)
        print(str(p)+'_'+str(q), ' :', best_scores[str(p)+'_'+str(q)])
    
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)


#%%
func = GetPersTropicalCoordinatesFeature
from fashionMNIST_tropical_optimisation import tropical_classifier

parameters = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(1,1000)},
    #{'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000), 'r': uniform(1,1000)}
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
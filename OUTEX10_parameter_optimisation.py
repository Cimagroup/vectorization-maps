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
from numpy.random import choice
from numpy.random import seed

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
#Methods with no parameter
func_list = [
             # GetPersStats,
             #GetCarlssonCoordinatesFeature
            ]

for func in func_list:
    print(func.__name__)
    
    with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
        features_l_d0 = pickle.load(f)
    with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
        features_l_d1 = pickle.load(f)
    with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
        features_u_d0 = pickle.load(f)
    with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
        features_u_d1 = pickle.load(f)
        
    param_grid = [
        {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
        {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450)},
        {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
        'gamma': expon(scale=.1)},
        {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
        'degree': [2,3], 'gamma': expon(scale=.1)},
    ]

    search =  RandomizedSearchCV(
        main_classifier(), param_distributions=param_grid, cv=5, n_iter=20,
        return_train_score=True, scoring='accuracy'
    )
    
    # search =  GridSearchCV(
    #     main_classifier(), param_grid=param_grid, cv=10,
    #     return_train_score=True, scoring='accuracy'
    # )
    
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
func_list = [
             #GetPersEntropyFeature,
             #GetBettiCurveFeature,
             #GetTopologicalVectorFeature,
             #GetPersLifespanFeature,
             #GetAtolFeature,
             GetPersImageFeature
            ]

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetPersImageFeature'] = [50,100,150,200,250]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetAtolFeature'] = [2,4,8,16]


for func in func_list:
    print(func.__name__)
    
    with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
        features_l_d0 = pickle.load(f)
    with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
        features_l_d1 = pickle.load(f)
    with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
        features_u_d0 = pickle.load(f)
    with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
        features_u_d1 = pickle.load(f)
        
    param_grid = [
         {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
         # {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450)},
         # {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
         #  'gamma': expon(scale=.1)},
         # {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
         #  'degree': [2,3], 'gamma': expon(scale=.1)},
    ]

    # search =  RandomizedSearchCV(
    #     main_classifier(), param_distributions=param_grid, cv=10, n_iter=40,
    #     return_train_score=True, scoring='accuracy'
    #  )
    search =  GridSearchCV(
        main_classifier(), param_grid=param_grid, cv=5,
        return_train_score=True, scoring='accuracy'
    )
    
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
func_list = [
             #GetPersSilhouetteFeature,
             #GetComplexPolynomialFeature,
             #GetPersLandscapeFeature
            ]

hyper_parameters = {}
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [1,2,3,5,10,20]]
hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]


for func in func_list:
    print(func.__name__)
    
    with open(path_feat + func.__name__ +'_l_d0.pkl', 'rb') as f:
        features_l_d0 = pickle.load(f)
    with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
        features_l_d1 = pickle.load(f)
    with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
        features_u_d0 = pickle.load(f)
    with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
        features_u_d1 = pickle.load(f)
        
    param_grid = [
        {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
         {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450)},
         {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
          'gamma': expon(scale=.1)},
         {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
          'degree': [2,3], 'gamma': expon(scale=.1)}
     ]
    search =  RandomizedSearchCV(
        main_classifier(), param_distributions=param_grid, cv=10, n_iter=40,
        return_train_score=True, scoring='accuracy'
    )
    # search =  GridSearchCV(
    # main_classifier(), param_grid=param_grid, cv=10, 
    # return_train_score=True, scoring='accuracy'
    # )
    
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
# func = GetPersTropicalCoordinatesFeature
# from OUTEX_tropical_optimisation import tropical_classifier

# param_grid = [
#     {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(0,500)}#,
#     #{'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450), 'r': uniform(0,500)},
#     #{'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
#     # 'gamma': expon(scale=.1), 'r': uniform(0,500)},
#     #{'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
#     # 'degree': [2,3], 'gamma': expon(scale=.1), 'r': uniform(0,500)},
# ]

# search =  RandomizedSearchCV(
#     tropical_classifier(), param_distributions=param_grid, cv=10, n_iter=40,
#     return_train_score=True, scoring='accuracy'
# )

# X_train, X_test = Z_train, Z_test
# search.fit(X_train, y_train)

# best_scores = (search.best_params_, search.best_score_)
# print(best_scores)
# with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
#   pickle.dump(best_scores, f)
      
#GetPersStats   
#({'base_estimator': 'RF', 'n_estimators': 300}, 0.9857142857142858)
#GetCarlssonCoordinates   
# ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9642857142857142)
#GetPersEntropyFeature
# 50  : ({'C': 271.87681709513566, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.9357142857142857)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.95)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.95)
#GetBettiCurveFeature
# 50  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9535714285714285)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9571428571428571)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.95)
#GetPersImages
# 50  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9035714285714285)
# 100  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9071428571428571)
# 150  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9214285714285715)
# 200  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9035714285714287)
# 250  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9)

#GetPersLifeSpan
# 50  : ({'C': 327.31825687032403, 'base_estimator': 'SVM', 'gamma': 0.07487827357893098, 'kernel': 'rbf'}, 0.975)
# 100  : ({'C': 588.6862732930688, 'base_estimator': 'SVM', 'gamma': 0.0339886976822587, 'kernel': 'rbf'}, 0.9678571428571429)
# 200  : ({'C': 937.5633724779192, 'base_estimator': 'SVM', 'gamma': 0.01959376913638965, 'kernel': 'rbf'}, 0.9714285714285715)
#GetAtolFeature
# 2  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8178571428571428)
# 4  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.6214285714285714)
# 8  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.575)
# 16  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.45357142857142857)
#GetPersSilhouette
# 50_1  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9785714285714286)
# 50_2  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9750000000000002)
# 50_3  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9785714285714286)
# 50_5  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9785714285714286)
# 50_10  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9750000000000002)
# 50_20  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9785714285714286)
# 100_1  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9750000000000002)
# 100_2  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9714285714285715)
# 100_3  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.975)
# 100_5  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9714285714285715)
# 100_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9714285714285715)
# 100_20  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9750000000000002)
# 200_1  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9714285714285715)
# 200_2  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9714285714285715)
# 200_3  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9714285714285715)
# 200_5  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9678571428571429)
# 200_10  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9714285714285715)
# 200_20  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9714285714285715)

#GetComplexPolynomial
# 5_R  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9607142857142857)
# 5_S  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8964285714285714)
# 5_T  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.39642857142857146)
# 10_R  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9607142857142856)
# 10_S  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9071428571428571)
# 10_T  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.4)
# 20_R  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9392857142857143)
# 20_S  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9214285714285714)
# 20_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.43928571428571433)

#Landscape
# 50_2  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9285714285714286)
# 50_5  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9428571428571428)
# 50_10  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9428571428571428)
# 50_20  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9535714285714286)
# 100_2  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9321428571428572)
# 100_5  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9428571428571427)
# 100_10  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9464285714285714)
# 100_20  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.9535714285714286)
# 200_2  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9321428571428572)
# 200_5  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9464285714285714)
# 200_10  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.9428571428571428)
# 200_20  : ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9571428571428571)
      


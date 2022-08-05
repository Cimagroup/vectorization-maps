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
#Methods with no parameter
func_list = [
             #GetPersStats,
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
        main_classifier(), param_distributions=param_grid, cv=5, n_iter=5,
        return_train_score=True, scoring='accuracy'
    )
    
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
             GetAtolFeature
             #GetPersImageFeature
            ]

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
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
         {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]}#,
        # {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450)},
        # {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
        #  'gamma': expon(scale=.1)},
        # {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
        #  'degree': [2,3], 'gamma': expon(scale=.1)},
    ]

    # search =  RandomizedSearchCV(
    #     main_classifier(), param_distributions=param_grid, cv=5, n_iter=20,
    #     return_train_score=True, scoring='accuracy'
    # )
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
        
    param_grid = [{'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]}]
    #     {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    #     {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450)},
    #     {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
    #      'gamma': expon(scale=.1)},
    #     {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
    #      'degree': [2,3], 'gamma': expon(scale=.1)},
    # ]

    # search =  RandomizedSearchCV(
    #     main_classifier(), param_distributions=param_grid, cv=5, n_iter=20,
    #     return_train_score=True, scoring='accuracy'
    # )
    search =  GridSearchCV(
    main_classifier(), param_grid=param_grid, cv=5, 
    return_train_score=True, scoring='accuracy'
    )
    
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
from tropical_optimisation import tropical_classifier

param_grid = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(0,500)}#,
    #{'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(50,450), 'r': uniform(0,500)},
    #{'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,999), 
    # 'gamma': expon(scale=.1), 'r': uniform(0,500)},
    #{'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,999), 
    # 'degree': [2,3], 'gamma': expon(scale=.1), 'r': uniform(0,500)},
]

search =  RandomizedSearchCV(
    tropical_classifier(), param_distributions=param_grid, cv=5, n_iter=20,
    return_train_score=True, scoring='accuracy'
)

X_train, X_test = Z_train, Z_test
search.fit(X_train, y_train)

best_scores = (search.best_params_, search.best_score_)
print(best_scores)
with open(path_feat + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
      
#GetPersStats   
# ({'base_estimator': 'RF', 'n_estimators': 500}, 0.9)
#GetCarlssonCoordinates   
# ({'base_estimator': 'RF', 'n_estimators': 500}, 0.84)
#GetPersEntropyFeature
#50  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7762577704102777)
#100  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7804613896947092)
#200  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7825611272275175)
#GetBettiCurveFeature
#50  : ({'C': 12.983791108944736, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.1133539803040063, 'kernel': 'poly'}, 0.7899184970299766)
#100  : ({'C': 105.0011705922286, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.008911616560745357, 'kernel': 'poly'}, 0.7967412626053323)
#200  : ({'C': 460.3875228825492, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7956872496201133)
#GetPersLifeSpan
#50  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.828780218262191)
#100  : ({'C': 211.40305469274247, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8314076529907446)
#200  : ({'C': 43.0664206583443, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.07294561046785114, 'kernel': 'poly'}, 0.8282594280978035)
#GetAtolFeature
#GetPersSilhouette
#50_1  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8103895565685868)
# 50_2  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8198535709352119)
# 50_3  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.81879679513745)
# 50_5  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8235225859925404)
# 50_10  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8214269926785468)
# 50_20  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8214283740848183)
# 100_1  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.817749689183589)
# 100_2  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8188009393562646)
# 100_3  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8188023207625363)
# 100_5  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8266804807293825)
# 100_10  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.8240530460008288)
# 100_20  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.822474098632408)
# 200_1  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8235267302113553)
# 200_2  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.820375742505871)
# 200_3  : ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8224810056637658)
# 200_5  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8229976516093384)
# 200_10  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8235267302113553)
# 200_20  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8245793617903026)
#GetComplexPolynomial
#5_R  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7778339549661555)
#5_S  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7232172952065202)
#5_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.2274209144909518)
#10_R  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7725818483215914)
#10_S  : ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7352894046138969)
#10_T  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.2442367730349496)
#20_R  : ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7274236773034949)
#20_S  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7584003315375052)
#20_T  : ({'base_estimator': 'RF', 'n_estimators': 300}, 0.2699778974996546)
#Landscape
#50_2 : ({'C': 279.98826657235054, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.019169247422775145, 'kernel': 'poly'}, 0.7815209283050144)
#50_5  : ({'C': 389.36250053681135, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.806199751346871)
#50_10  : ({'C': 388.63916688339515, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8214352811161764)
#50_20  : ({'C': 69.36936891467181, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8308882442326289)
#100_2  : ({'C': 319.22789736628494, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.27657307946979176, 'kernel': 'poly'}, 0.7825721784776902)
#100_5  : ({'C': 63.47999640983946, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8046235667909933)
#100_10  : ({'C': 249.50076420380535, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8214311368973615)
#100_20  : ({'C': 97.99352629982499, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8287871252935488)
#200_2  : ({'C': 6.564283095165183, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.027056316626459177, 'kernel': 'poly'}, 0.7815223097112861)
#200_5  : ({'C': 217.27436700818845, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.806199751346871)
#200_10  : ({'C': 230.78002905938382, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8209048211078878)
#200_20  : ({'C': 166.60500600800455, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8287871252935488)

      


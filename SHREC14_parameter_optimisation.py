import pickle
import pandas as pd
from vectorisation import *
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from direct_optimisation import main_classifier
from scipy.stats import uniform
from scipy.stats import expon
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


data_path = 'Shrec14/data/'
feat_path = "Shrec14/features/"
data = pd.read_csv(data_path+'Uli_data.csv')

#%%
#Load the barcodes

def reshapeVec(g):
    A = np.array([g.dim,g.birth,g.death])
    A = A.T
    return A

def getDgm(A, dim = 0):
    if type(dim) != str:
        if dim == 0:
            A = A[np.where(np.logical_or(A[:,0] == dim, A[:,0] == -1))[0],1:]
            
        if dim == 1:
            A = A[np.where(np.logical_or(A[:,0] == dim, A[:,0] == -2))[0],1:]
    
    return(A)

dgmsDF = data.groupby(['freq', 'trial']).apply(reshapeVec)
dgmsDF = dgmsDF.reset_index()
dgmsDF = dgmsDF.rename(columns = {0:'CollectedDgm'})
dgmsDF['Dgm1'] = dgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 1))
dgmsDF['Dgm0'] = dgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 0))
dgmsDF['DgmInf'] = dgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 'essential'))


#%%
#Define the labels

def label(index):
    if 0 <= index <= 19:
        return 'male_neutral'
    elif 20<= index <=39:
        return 'male_bodybuilder'
    elif 40<= index <=59:
        return 'male_fat'
    elif 60<= index <=79:
        return 'male_thin'
    elif 80<= index <=99:
        return 'male_average'
    elif 100<= index <=119:
        return 'female_neutral'
    elif 120<= index <=139:
        return 'female_bodybuilder'
    elif 140<= index <=159:
        return 'female_fat'
    elif 160<= index <=179:
        return 'female_thin'
    elif 180<= index <=199:
        return 'female_average'
    elif 200<= index <=219:
        return 'child_neutral'
    elif 220<= index <=239:
        return 'child_bodybuilder'
    elif 240<= index <=259:
        return 'child_fat'
    elif 260<= index <=279:
        return 'child_thin'
    elif 280<= index <=299:
        return 'child_average'
    else:
        print('What are you giving me?')
        
dgmsDF['TrainingLabel'] = dgmsDF.freq.apply(label)
dgmsDF= dgmsDF.sample(frac=1)

labelsD = {}
Z_train = {}
Z_test = {}
y_train = {}
y_test = {}
#The order of the diagrams change for each t, then we cannot use the same labels
#for each t. We decide to create an split for every t.
for t in range(1,11):
    T = dgmsDF[dgmsDF.trial==t]
    
    labels = np.array(T['TrainingLabel'])
    labels = pd.DataFrame(labels)
    label_names = labels.copy()
    
    label_unique = pd.DataFrame(labels)
    label_unique = label_unique[0].unique()
    
    i=0
    for l in label_unique:
        labels[labels == l]=i
        i += 1
    
    labelsD[str(t)] = labels[0].tolist()
    label_names[str(t)] = label_names[0].tolist()

#%%
func_list = [
             # GetPersStats,
             GetCarlssonCoordinatesFeature
            ]

param_grid = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    # {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    # {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
    # 'gamma': expon(scale=.1)},
    # {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
    # 'degree': [2,3], 'gamma': expon(scale=.1)},
]

# search =  RandomizedSearchCV(
#     main_classifier(), param_distributions=param_grid, cv=10, n_iter=40,
#     return_train_score=True, scoring='accuracy'
# )

search =  GridSearchCV(
    main_classifier(), param_grid=param_grid, cv=10,
    return_train_score=True, scoring='accuracy'
)
        
for func in func_list:
    print(func.__name__)
    best_scores = {}
    for t in range(1,11):
        print(t)
        X = []
        dgms = dgmsDF[dgmsDF.trial==t]
        dgms = np.array(dgms['Dgm1'])
        for i in range(dgms.shape[0]):
            X.append(func(dgms[i]))          
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            labelsD[str(t)], 
                                                            test_size=0.3, 
                                                            random_state=t)

        search.fit(X_train, y_train) 

        best_scores[str(t)] = (search.best_params_, search.best_score_)
        
    print(best_scores)
    with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
      
#%%
func_list = [
             GetPersEntropyFeature,
             GetBettiCurveFeature,
             #GetTopologicalVectorFeature,
             GetPersLifespanFeature,
             #GetAtolFeature,
             # GetPersImageFeature
            ]


hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetPersImageFeature'] = [10,25,50,100,200]
hyper_parameters['GetAtolFeature'] = [2,4,8]

param_grid = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
    'gamma': expon(scale=.1)},
    {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
    'degree': [2,3], 'gamma': expon(scale=.1)},
]

search =  RandomizedSearchCV(
    main_classifier(), param_distributions=param_grid, cv=10, n_iter=40,
    return_train_score=True, scoring='accuracy'
)

# search =  GridSearchCV(
#     main_classifier(), param_grid=param_grid, cv=10,
#     return_train_score=True, scoring='accuracy'
# )
        
for func in func_list:
    print(func.__name__)
    best_scores = {}

    for t in range(1,11):
        for p in hyper_parameters[func.__name__]:
            print(t,p)
            X = []
            dgms = dgmsDF[dgmsDF.trial==t]
            dgms = np.array(dgms['Dgm1'])
            for i in range(dgms.shape[0]):
                X.append(func(dgms[i],p))          
            X_train, X_test, y_train, y_test = train_test_split(X, 
                                                                labelsD[str(t)], 
                                                                test_size=0.3, 
                                                                random_state=t)
    
            search.fit(X_train, y_train) 
    
            best_scores[str(t)+'_'+str(p)] = (search.best_params_, search.best_score_)
        
    print(best_scores)
    with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
      pickle.dump(best_scores, f)
      
          
          
# GetPersStats    
# {'1': ({'C': 242.61602391361458, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7809523809523811), 
# '2': ({'C': 34.41327954736772, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.6666666666666666), 
# '3': ({'C': 327.14773416776956, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.6761904761904761), 
# '4': ({'C': 75.87960668267124, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.738095238095238), 
# '5': ({'C': 6.063748502539767, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8571428571428571), 
# '6': ({'C': 547.0104892185311, 'base_estimator': 'SVM', 'gamma': 0.0022931176685105576, 'kernel': 'rbf'}, 0.8523809523809524), 
# '7': ({'C': 277.311870239477, 'base_estimator': 'SVM', 'gamma': 0.005724516397732078, 'kernel': 'rbf'}, 0.9285714285714286), 
# '8': ({'C': 771.0945390318818, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0180808489464352, 'kernel': 'poly'}, 0.8761904761904761), 
# '9': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8857142857142856), 
# '10': ({'C': 219.44680984683285, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.919047619047619)}
# GetCarlssonCoordinatesFeature
# {'1': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.523809523809524), 
#  '2': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.3666666666666667), 
#  '3': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.4714285714285714), 
#  '4': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7285714285714286), 
#  '5': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8666666666666666), 
#  '6': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9142857142857143), 
#  '7': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9095238095238096), 
#  '8': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.7666666666666667), 
#  '9': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.719047619047619), 
#  '10': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.6952380952380952)}
        

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
             # GetCarlssonCoordinatesFeature
            ]

param_grid = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    # {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    # {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
    # 'gamma': expon(scale=.01)},
    # {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
    # 'degree': [2,3], 'gamma': expon(scale=.01)},
]

# search =  RandomizedSearchCV(
#     main_classifier(), param_distributions=param_grid, cv=5, n_iter=40,
#     return_train_score=True, scoring='accuracy'
# )

search =  GridSearchCV(
    main_classifier(), param_grid=param_grid, cv=5,
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
             # GetPersEntropyFeature,
             # GetBettiCurveFeature,
             #GetTopologicalVectorFeature,o
             # GetPersLifespanFeature,
             #GetAtolFeature,
             GetPersImageFeature
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
    # {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
    'gamma': expon(scale=.01)},
    # {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
    # 'degree': [2,3], 'gamma': expon(scale=.01)},
]

search =  RandomizedSearchCV(
    main_classifier(), param_distributions=param_grid, cv=5, n_iter=40,
    return_train_score=True, scoring='accuracy'
)

# search =  GridSearchCV(
#     main_classifier(), param_grid=param_grid, cv=5,
#     return_train_score=True, scoring='accuracy'
# )
        
for func in func_list:
    print(func.__name__)
    best_scores = {}

    for t in range(9,11):#range(1,11):
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
            print(best_scores[str(t)+'_'+str(p)])
        
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
# GetPersEntropyFeature
# {'1_50': ({'C': 316.968132174157, 'base_estimator': 'SVM', 'gamma': 0.010308558896667146, 'kernel': 'rbf'}, 0.4714285714285714), 
# '1_100': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.4619047619047619), 
# '1_200': ({'C': 333.73240112817837, 'base_estimator': 'SVM', 'gamma': 0.004816329290999086, 'kernel': 'rbf'}, 0.49047ba61904761905), 
# '2_50': ({'C': 713.4869982253656, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.007728077520717372, 'kernel': 'poly'}, 0.3857142857142858), 
# '2_100': ({'C': 575.1996299088638, 'base_estimator': 'SVM', 'gamma': 0.009220910833683411, 'kernel': 'rbf'}, 0.42857142857142855), 
# '2_200': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.4333333333333334), 
# '3_50': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.4333333333333334), 
# '3_100': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.4428571428571429), 
# '3_200': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.4619047619047619), 
# '4_50': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.5904761904761905), 
# '4_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.6047619047619047), 
# '4_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.6428571428571429), 
# '5_50': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.6761904761904762), 
# '5_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7), 
# '5_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7190476190476189), 
# '6_50': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.6476190476190476), 
# '6_100': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7), 
# '6_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7142857142857144), 
# '7_50': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.6476190476190476), 
# '7_100': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.6714285714285715), 
# '7_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7142857142857143), 
# '8_50': ({'C': 64.34064203208412, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.011513553368356542, 'kernel': 'poly'}, 0.619047619047619), 
# '8_100': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.6047619047619047), 
# '8_200': ({'C': 137.5518127520976, 'base_estimator': 'SVM', 'gamma': 0.001966757624327403, 'kernel': 'rbf'}, 0.6571428571428571), 
# '9_50': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.5285714285714286), 
# '9_100': ({'C': 167.17443031566526, 'base_estimator': 'SVM', 'gamma': 0.0029046626574237167, 'kernel': 'rbf'}, 0.5666666666666667), 
# '9_200': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.5666666666666667), 
# '10_50': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.4238095238095238), 
# '10_100': ({'C': 236.75530013409286, 'base_estimator': 'SVM', 'gamma': 0.004904869721469778, 'kernel': 'rbf'}, 0.44761904761904764), 
# '10_200': ({'C': 39.3728734046983, 'base_estimator': 'SVM', 'gamma': 0.008857593983137213, 'kernel': 'rbf'}, 0.43809523809523815)}
# GetBettiCurveFeature
# {'1_50': ({'C': 979.843633584525, 'base_estimator': 'SVM', 'gamma': 0.00015556473024832092, 'kernel': 'rbf'}, 0.5666666666666667), 
#  '1_100': ({'C': 604.8218743986274, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.005052751125437617, 'kernel': 'poly'}, 0.6238095238095238), 
#  '1_200': ({'C': 685.1757925863737, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.026095092746128038, 'kernel': 'poly'}, 0.6333333333333334), 
#  '2_50': ({'C': 450.4988335857304, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.5047619047619047), 
#  '2_100': ({'C': 707.6263512818409, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.010457978529920224, 'kernel': 'poly'}, 0.6238095238095237), 
#  '2_200': ({'C': 328.02019732229604, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.0005981801455743634, 'kernel': 'poly'}, 0.6095238095238095), 
#  '3_50': ({'C': 744.8821588325877, 'base_estimator': 'SVM', 'gamma': 6.946505020893618e-05, 'kernel': 'rbf'}, 0.5904761904761905), 
#  '3_100': ({'C': 206.01770354273862, 'base_estimator': 'SVM', 'gamma': 0.0030882237830110715, 'kernel': 'rbf'}, 0.6857142857142857), 
#  '3_200': ({'C': 638.5460549526342, 'base_estimator': 'SVM', 'gamma': 0.001775979433833502, 'kernel': 'rbf'}, 0.680952380952381), 
#  '4_50': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.680952380952381), 
#  '4_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7047619047619047), 
#  '4_200': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.7285714285714285), 
#  '5_50': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.680952380952381), 
#  '5_100': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.7428571428571428), 
#  '5_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7714285714285715), 
#  '6_50': ({'C': 235.95766515906215, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0002888598356868032, 'kernel': 'poly'}, 0.5428571428571429), 
#  '6_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.6333333333333333), 
#  '6_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.6428571428571428), 
#  '7_50': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.5904761904761904), 
#  '7_100': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.6190476190476191), 
#  '7_200': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.6476190476190475), 
#  '8_50': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.5285714285714286), 
#  '8_100': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.5809523809523809), 
#  '8_200': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.6), 
#  '9_50': ({'C': 601.0329172783324, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.03597668190196009, 'kernel': 'poly'}, 0.4333333333333333), 
#  '9_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.5285714285714286), 
#  '9_200': ({'C': 45.91396006997594, 'base_estimator': 'SVM', 'gamma': 0.0002056231610313888, 'kernel': 'rbf'}, 0.5761904761904761), 
#  '10_50': ({'C': 419.1008064610391, 'base_estimator': 'SVM', 'gamma': 0.0010490816605971444, 'kernel': 'rbf'}, 0.43809523809523804), 
#  '10_100': ({'C': 431.1734558762402, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0009102852508035878, 'kernel': 'poly'}, 0.4428571428571428), 
#  '10_200': ({'C': 838.3106508517023, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.03928745645869667, 'kernel': 'poly'}, 0.4523809523809524)}
# GetPersLifeSpanFeature
# {'1_50': ({'C': 503.21308250745943, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.004891342877003531, 'kernel': 'poly'}, 0.49523809523809514), 
#  '1_100': ({'C': 372.6342339205362, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.003513059718027979, 'kernel': 'poly'}, 0.5095238095238095), 
#  '1_200': ({'C': 879.5184108555763, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0349498757941561, 'kernel': 'poly'}, 0.5142857142857142), 
#  '2_50': ({'C': 620.3777502107488, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.010556597926643768, 'kernel': 'poly'}, 0.4619047619047619), 
#  '2_100': ({'C': 770.5132102922089, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.4666666666666666), 
#  '2_200': ({'C': 775.2915571850108, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.48571428571428565), 
#  '3_50': ({'C': 402.2480693852811, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.015306820080996165, 'kernel': 'poly'}, 0.5476190476190477), 
#  '3_100': ({'C': 934.7478921569633, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.5761904761904763), 
#  '3_200': ({'C': 205.51602581712015, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.5857142857142857), 
#  '4_50': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.7047619047619047), 
#  '4_100': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7428571428571429), 
#  '4_200': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.7333333333333333), 
#  '5_50': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.8238095238095239), 
#  '5_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.8142857142857144), 
#  '5_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8238095238095238), 
#  '6_50': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8095238095238095), 
#  '6_100': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.8476190476190476), 
#  '6_200': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.8428571428571429), 
#  '7_50': ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8476190476190476), 
#  '7_100': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.8523809523809524), 
#  '7_200': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8619047619047621), 
#  '8_50': ({'C': 42.17778132301109, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7761904761904762), 
#  '8_100': ({'C': 749.1066776206693, 'base_estimator': 'SVM', 'gamma': 0.0004597252123264237, 'kernel': 'rbf'}, 0.8), 
#  '8_200': ({'C': 809.5469892495846, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8238095238095239), 
#  '9_50': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7238095238095239), 
#  '9_100': ({'C': 72.26567291369568, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7571428571428571), 
#  '9_200': ({'C': 142.0011634426691, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7761904761904762), 
#  '10_50': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.5761904761904761), 
#  '10_100': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.5857142857142856), 
#  '10_200': ({'C': 493.4855615116065, 'base_estimator': 'SVM', 'degree': 2, 'gamma': 0.0023140880441517193, 'kernel': 'poly'}, 0.5857142857142857)}
# PersImages
# 1 10 ({'base_estimator': 'RF', 'n_estimators': 100}, 0.5571428571428572)
# 1 25 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.6904761904761905)
# 1 50 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7238095238095239)
# 1 100 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7571428571428571)
# 1 200 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.761904761904762)
# 2 10 ({'base_estimator': 'RF', 'n_estimators': 200}, 0.5904761904761905)
# 2 25 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.6952380952380952)
# 2 50 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.738095238095238)
# 2 100 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7238095238095237)
# 2 200 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.738095238095238)
# 3 10 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7142857142857142)
# 3 25 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8238095238095238)
# 3 50 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8428571428571429)
# 3 100 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8333333333333334)
# 3 200 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8476190476190476)
# 4 10 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8285714285714285)
# 4 25 ({'base_estimator': 'RF', 'n_estimators': 200}, 0.8571428571428571)
# 4 50 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.8571428571428571)
# 4 100 ({'C': 314.2287117683795, 'base_estimator': 'SVM', 'gamma': 0.0016111996515498679, 'kernel': 'rbf'}, 0.8476190476190476)
# 4 200 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8523809523809524)
# 5 10 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.8238095238095239)
# 5 25 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8333333333333334)
# 5 50 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.8476190476190476)
# 5 100 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8238095238095239)
# 5 200 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8238095238095239)
# 6 10 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8904761904761905)
# 6 25 ({'base_estimator': 'RF', 'n_estimators': 200}, 0.8904761904761905)
# 6 50 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8857142857142858)
# 6 100 ({'base_estimator': 'RF', 'n_estimators': 300}, 0.880952380952381)
# 6 200 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.8714285714285713)
# 7 10 ({'C': 843.1950332591473, 'base_estimator': 'SVM', 'gamma': 0.006160874043708344, 'kernel': 'rbf'}, 0.8714285714285713)
# 7 25 ({'C': 230.1488614176149, 'base_estimator': 'SVM', 'gamma': 0.0057641870095165235, 'kernel': 'rbf'}, 0.8809523809523808)
# 7 50 ({'C': 500.09325693790595, 'base_estimator': 'SVM', 'gamma': 0.0017326609098863815, 'kernel': 'rbf'}, 0.8809523809523808)
# 7 100 ({'C': 108.15265139116536, 'base_estimator': 'SVM', 'gamma': 0.0007232398939433436, 'kernel': 'rbf'}, 0.8857142857142856)
# 7 200 ({'C': 53.93419560916834, 'base_estimator': 'SVM', 'gamma': 0.0004992661260813728, 'kernel': 'rbf'}, 0.8857142857142858)
# 8 10 ({'C': 360.00349917907647, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7904761904761906)
# 8 25 ({'C': 47.172950096021296, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7904761904761904)
# 8 50 ({'C': 702.5704546364792, 'base_estimator': 'SVM', 'gamma': 0.002037217007460523, 'kernel': 'rbf'}, 0.7761904761904763)
# 8 100 ({'C': 984.0333148161211, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.7666666666666667)
# 8 200 ({'C': 130.86586061254002, 'base_estimator': 'SVM', 'degree': 3, 'gamma': 0.00029743964172875204, 'kernel': 'poly'}, 0.7714285714285715)
# 9 10 ({'base_estimator': 'RF', 'n_estimators': 200}, 0.6285714285714287)
# 9 25 ({'base_estimator': 'RF', 'n_estimators': 200}, 0.6380952380952382)
# 9 50 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.6285714285714284)
# 9 100 ({'base_estimator': 'RF', 'n_estimators': 500}, 0.6333333333333334)
# 9 200 ({'base_estimator': 'RF', 'n_estimators': 200}, 0.6333333333333334)
# 10 10 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.44761904761904764)
# 10 25 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.4666666666666667)
# 10 50 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.4523809523809524)
# 10 100 ({'base_estimator': 'RF', 'n_estimators': 100}, 0.44761904761904764)
# 10 200 ({'base_estimator': 'RF', 'n_estimators': 50}, 0.44761904761904764)
  

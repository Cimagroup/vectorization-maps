import pickle
import pandas as pd
from vectorisation import *
import numpy as np
from numpy.random import seed
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from direct_optimisation import main_classifier
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
#hyperparameters from the methods
hyper_parameters = {}

hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetPersImageFeature'] = [10,25,50,100,200]
hyper_parameters['GetAtolFeature'] = [2,4,8]
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]


#%%

func = GetPersStats

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    print(t)
    X_train = []
    for i in Z_train[str(t)]:       
        X_train.append(features[str(t)+'_'+str(i)])
    
    search = searchR(complete)
    search.fit(X_train, y_train[str(t)]) 

    best_scores[str(t)] = (search.best_params_, search.best_score_)
    print(best_scores[str(t)])
    
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
#%%

func = GetCarlssonCoordinatesFeature
print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    print(t)
    X_train = []
    for i in Z_train[str(t)]:       
        X_train.append(features[str(t)+'_'+str(i)])
    
    search = searchG(onlyForest)
    search.fit(X_train, y_train[str(t)]) 

    best_scores[str(t)] = (search.best_params_, search.best_score_)
    print(best_scores[str(t)])
    
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func=GetPersEntropyFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__]:
        print(t,p)
        X_train = []
        for i in Z_train[str(t)]:       
            X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
        
        search = searchR(complete)
        search.fit(X_train, y_train[str(t)])
        
        if search.best_score_ > best_scores[str(t)][0][2]:
            best_scores[str(t)] = [('p='+str(p),search.best_params_, search.best_score_)]
        elif search.best_score_ == best_scores[str(t)][0][2]:
            best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p),search.best_params_, search.best_score_)]

    print(best_scores[str(t)])
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
#%%
func=GetBettiCurveFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__]:
        print(t,p)
        X_train = []
        for i in Z_train[str(t)]:       
            X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
        
        if t<10:
            search = searchR(complete)
        else:
            search = searchG(onlyForest)
        search.fit(X_train, y_train[str(t)])
        
        if search.best_score_ > best_scores[str(t)][0][2]:
            best_scores[str(t)] = [('p='+str(p),search.best_params_, search.best_score_)]
        elif search.best_score_ == best_scores[str(t)][0][2]:
            best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p),search.best_params_, search.best_score_)]

    print(best_scores[str(t)])
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func=GetPersLifespanFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__]:
        print(t,p)
        X_train = []
        for i in Z_train[str(t)]:       
            X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
        
        search = searchR(complete)
        search.fit(X_train, y_train[str(t)])
        
        if search.best_score_ > best_scores[str(t)][0][2]:
            best_scores[str(t)] = [('p='+str(p),search.best_params_, search.best_score_)]
        elif search.best_score_ == best_scores[str(t)][0][2]:
            best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p),search.best_params_, search.best_score_)]

    print(best_scores[str(t)])
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
#%%

func=GetTopologicalVectorFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__]:
        print(t,p)
        X_train = []
        for i in Z_train[str(t)]:       
            X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
        
        search = searchG(onlyForest)
        search.fit(X_train, y_train[str(t)])
        
        if search.best_score_ > best_scores[str(t)][0][2]:
            best_scores[str(t)] = [('p='+str(p),search.best_params_, search.best_score_)]
        elif search.best_score_ == best_scores[str(t)][0][2]:
            best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p),search.best_params_, search.best_score_)]

    print(best_scores[str(t)])
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%
func=GetPersImageFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__]:
        print(t,p)
        X_train = []
        for i in Z_train[str(t)]:       
            X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
        
        if t<9:
            search = searchR(complete)
        else:
            search = searchR(forestRBF)
        search.fit(X_train, y_train[str(t)])
        
        if search.best_score_ > best_scores[str(t)][0][2]:
            best_scores[str(t)] = [('p='+str(p),search.best_params_, search.best_score_)]
        elif search.best_score_ == best_scores[str(t)][0][2]:
            best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p),search.best_params_, search.best_score_)]

    print(best_scores[str(t)])
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)

#%%Atol

func=GetAtolFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,10):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__]:
        print(t,p)
        X_train = []
        for i in Z_train[str(t)]:       
            X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
        if t >= 4:
            search = searchR(noPoly)
        else:
            search = searchR(forestRBF)
        search.fit(X_train, y_train[str(t)])
        
        if search.best_score_ > best_scores[str(t)][0][2]:
            best_scores[str(t)] = [('p='+str(p),search.best_params_, search.best_score_)]
        elif search.best_score_ == best_scores[str(t)][0][2]:
            best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p),search.best_params_, search.best_score_)]

    print(best_scores[str(t)])
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)
  
#%%
func = GetPersSilhouetteFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print('t,p,q = ', t, p, q)
            X_train = []
            for i in Z_train[str(t)]:
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
            search = searchR(noPoly)
            search.fit(X_train, y_train[str(t)])
            if search.best_score_ > best_scores[str(t)][0][2]:
                best_scores[str(t)] = [('p='+str(p)+'  q='+str(q), search.best_params_, search.best_score_)]
            elif search.best_score_ == best_scores[str(t)][0][2]:
                best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p)+'  q='+str(q), search.best_params_, search.best_score_)]
    print(best_scores[str(t)])
        
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
    pickle.dump(best_scores, f)  
    
#%%
func = GetComplexPolynomialFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print('t,p,q = ', t, p, q)
            X_train = []
            for i in Z_train[str(t)]:
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
                X_train=float64to32(X_train) 
            search = searchG(onlyForest)
            search.fit(X_train, y_train[str(t)])
            if search.best_score_ > best_scores[str(t)][0][2]:
                best_scores[str(t)] = [('p='+str(p)+'  q='+str(q), search.best_params_, search.best_score_)]
            elif search.best_score_ == best_scores[str(t)][0][2]:
                best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p)+'  q='+str(q), search.best_params_, search.best_score_)]
    print(best_scores[str(t)])
        
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
    pickle.dump(best_scores, f)
 
#%%
func = GetPersLandscapeFeature

print(func.__name__)
with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
    features = pickle.load(f)
best_scores = {}
for t in range(1,11):
    best_scores[str(t)]=[('a','a',0)]
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print('t,p,q = ', t, p, q)
            X_train = []
            for i in Z_train[str(t)]:
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
            if q!=2:
                search = searchR(complete)
            else:
                search = searchR(forestRBF)
            search.fit(X_train, y_train[str(t)])
            if search.best_score_ > best_scores[str(t)][0][2]:
                best_scores[str(t)] = [('p='+str(p)+'  q='+str(q), search.best_params_, search.best_score_)]
            elif search.best_score_ == best_scores[str(t)][0][2]:
                best_scores[str(t)] = best_scores[str(t)]+[('p='+str(p)+'  q='+str(q), search.best_params_, search.best_score_)]
    print(best_scores[str(t)])
        
print(best_scores)
with open(feat_path + func.__name__ + '_hyperparameter.pkl', 'wb') as f:
    pickle.dump(best_scores, f)  
         
#%%
func = GetPersTropicalCoordinatesFeature
from SHREC14_tropical_optimisation import tropical_classifier

best_scores = {}

param_grid = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500], 'r': uniform(0,200)},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000), 'r': uniform(0,200)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01), 'r': uniform(0,200)}
 ]
print(func.__name__)
for t in range(1,11):
    print(t)
    search =  RandomizedSearchCV(
        tropical_classifier(dgmsT=dgmsT, t=t), param_distributions=param_grid, cv=5, n_iter=40,
        return_train_score=True, scoring='accuracy', random_state=1
    )
    search.fit(Z_train[str(t)], y_train[str(t)])
    best_scores[str(t)] = (search.best_params_, search.best_score_)
    print(best_scores[str(t)])
    
print(best_scores)
with open(feat_path + 'GetTropicalCoordinatesFeature' + '_hyperparameter.pkl', 'wb') as f:
  pickle.dump(best_scores, f)          
 
#%%          
# '1': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.7571428571428571),
# '2': ({'C': 1000.0405153241447,'base_estimator': 'SVM','degree': 2,'gamma': 0.0009688387165373345, 'kernel': 'poly'},0.6476190476190476),
# '3': ({'C': 141.38693859523377, 'base_estimator': 'SVM', 'kernel': 'linear'},0.6476190476190475),
# '4': ({'C': 141.38693859523377, 'base_estimator': 'SVM', 'kernel': 'linear'},0.761904761904762),
# '5': ({'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.8),
# '6': ({'C': 288.77533858634877, 'base_estimator': 'SVM', 'gamma': 0.001392949093880637, 'kernel': 'rbf'}, 0.9),
# '7': ({'C': 936.5390708060319,'base_estimator': 'SVM','gamma': 0.01872823656893796,'kernel': 'rbf'},0.9047619047619048),
# '8': ({'C': 803.7575039373648, 'base_estimator': 'SVM', 'kernel': 'linear'},0.8666666666666666),
# '9': ({'C': 686.2195003967595, 'base_estimator': 'SVM', 'kernel': 'linear'},0.9142857142857144),
# '10': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.9238095238095239)}
#GetCarlssonCoordinates
# '1': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.4666666666666666),
# '2': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.3619047619047619),
# '3': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.45714285714285713),
# '4': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7238095238095238),
# '5': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.8571428571428571),
# '6': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.8857142857142858),
# '7': ({'base_estimator': 'RF', 'n_estimators': 50}, 0.9190476190476191),
# '8': ({'base_estimator': 'RF', 'n_estimators': 300}, 0.7714285714285715),
# '9': ({'base_estimator': 'RF', 'n_estimators': 100}, 0.7285714285714286),
# '10': ({'base_estimator': 'RF', 'n_estimators': 200}, 0.6714285714285715)}
#GetPersEntropy
# '1': [('p=200', {'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.4904761904761905)],
# '2': [('p=100',   {'C': 223.1245475353748, 'base_estimator': 'SVM','degree': 3,'gamma': 0.005940143639429267,'kernel': 'poly'},0.4809523809523809)],
# '3': [('p=200',{'base_estimator': 'RF', 'n_estimators': 300}, 0.4428571428571429)],
# '4': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100}, 0.6142857142857142)], 
# '5': [('p=100',{'base_estimator': 'RF', 'n_estimators': 100}, 0.680952380952381)],
# '6': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100}, 0.7047619047619047)],
# '7': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100}, 0.6904761904761905)],
# '8': [('p=200',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5761904761904763)],
# '9': [('p=50',{'C': 879.1425034294132, 'base_estimator': 'SVM', 'gamma': 0.0010352534930954075,'kernel': 'rbf'},0.5666666666666667)],
# '10': [('p=200',{'base_estimator': 'RF', 'n_estimators': 500},0.4809523809523809)]}
#GetBettiCurve
# '1': [('p=200',   {'C': 835.625671897373,'base_estimator': 'SVM','gamma': 0.00018457575175565604,'kernel': 'rbf'},0.7142857142857143)],
# '2': [('p=200',{'C': 288.77533858634877,'base_estimator': 'SVM','gamma': 0.001392949093880637,'kernel': 'rbf'},0.6095238095238095)],
# '3': [('p=100',{'C': 288.77533858634877,'base_estimator': 'SVM','gamma': 0.001392949093880637,'kernel': 'rbf'},0.7),
#   ('p=200',{'C': 288.77533858634877,'base_estimator': 'SVM','gamma': 0.001392949093880637,'kernel': 'rbf'},0.7)],
# '4': [('p=200',{'base_estimator': 'RF', 'n_estimators': 300},0.7523809523809524)],
# '5': [('p=200',{'base_estimator': 'RF', 'n_estimators': 200},0.7428571428571429)], 
# '6': [('p=200',{'base_estimator': 'RF', 'n_estimators': 300},0.6857142857142857)],
# '7': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100},0.6761904761904762)], 
# '8': [('p=200',{'base_estimator': 'RF', 'n_estimators': 500},0.5714285714285714)], 
# '9': [('p=100',{'base_estimator': 'RF', 'n_estimators': 500},0.5857142857142857),
#   ('p=200',{'base_estimator': 'RF', 'n_estimators': 300},0.5857142857142857)],
# '10': [('p=50',{'base_estimator': 'RF', 'n_estimators': 300},0.4333333333333333)]}
#GetPersLifespanFeature
# '1': [('p=200',{'C': 1000.0405153241447,'base_estimator': 'SVM','degree': 2,'gamma': 0.0009688387165373345,'kernel': 'poly'},0.5523809523809524)],
# '2': [('p=100',{'C': 1000.0405153241447,'base_estimator': 'SVM','degree': 2,'gamma': 0.0009688387165373345,'kernel': 'poly'},0.5238095238095238)], 
# '3': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100},0.5380952380952382)],
# '4': [('p=200',{'base_estimator': 'RF', 'n_estimators': 500},0.7333333333333333)],
# '5': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100},0.8142857142857144)], 
# '6': [('p=100',{'base_estimator': 'RF', 'n_estimators': 200},0.8666666666666668)],
# '7': [('p=200',{'C': 835.625671897373,'base_estimator': 'SVM','gamma': 0.00018457575175565604,'kernel': 'rbf'},0.9095238095238095)],
# '8': [('p=50',{'C': 835.625671897373,'base_estimator': 'SVM','gamma': 0.00018457575175565604,'kernel': 'rbf'},0.8142857142857143)],
# '9': [('p=100',{'C': 141.38693859523377, 'base_estimator': 'SVM', 'kernel': 'linear'},0.780952380952381)], 
# '10': [('p=200',{'base_estimator': 'RF', 'n_estimators': 50},0.5523809523809524)]}
#GetTopologicalVectorFeature
# {'1': [('p=20', {'base_estimator': 'RF', 'n_estimators': 100}, 0.40476190476190477)], 
#  '2': [('p=20', {'base_estimator': 'RF', 'n_estimators': 200}, 0.27142857142857146)], 
#  '3': [('p=20', {'base_estimator': 'RF', 'n_estimators': 50}, 0.29523809523809524)], 
#  '4': [('p=20', {'base_estimator': 'RF', 'n_estimators': 300}, 0.40476190476190477)], 
#  '5': [('p=20', {'base_estimator': 'RF', 'n_estimators': 500}, 0.5904761904761904)], 
#  '6': [('p=20', {'base_estimator': 'RF', 'n_estimators': 100}, 0.6428571428571429)], 
#  '7': [('p=20', {'base_estimator': 'RF', 'n_estimators': 200}, 0.7428571428571429)], 
#  '8': [('p=20', {'base_estimator': 'RF', 'n_estimators': 100}, 0.7285714285714286)], 
#  '9': [('p=20', {'base_estimator': 'RF', 'n_estimators': 100}, 0.6666666666666666)], 
#  '10': [('p=20', {'base_estimator': 'RF', 'n_estimators': 100}, 0.4714285714285714)]}
#GetPersImageFeature
# '1': [('p=200',{'base_estimator': 'RF', 'n_estimators': 300},0.7714285714285714)],
# '2': [('p=100',{'base_estimator': 'RF', 'n_estimators': 100},0.7952380952380953)],
# '3': [('p=100',{'base_estimator': 'RF', 'n_estimators': 500},0.8428571428571429)],
# '4': [('p=100',{'base_estimator': 'RF', 'n_estimators': 500},0.8952380952380953)],
# '5': [('p=50',{'base_estimator': 'RF', 'n_estimators': 50},0.8666666666666666)],
# '6': [('p=10',{'base_estimator': 'RF', 'n_estimators': 200},0.8857142857142858)],
# '7': [('p=200',{'C': 835.625671897373,'base_estimator': 'SVM','gamma': 0.00018457575175565604,'kernel': 'rbf'},0.919047619047619)],
# '8': [('p=10',{'C': 879.1425034294132,'base_estimator': 'SVM','gamma': 0.0010352534930954075,'kernel': 'rbf'},0.7904761904761906),
#  ('p=25',{'C': 835.625671897373,'base_estimator': 'SVM','gamma': 0.00018457575175565604,'kernel': 'rbf'},0.7904761904761906)],
# '9': [('p=25',{'base_estimator': 'RF', 'n_estimators': 100},0.6714285714285715),
#       ('p=200',{'base_estimator': 'RF', 'n_estimators': 300},0.6714285714285715)],
# '10': [('p=200',{'base_estimator': 'RF', 'n_estimators': 100},0.47619047619047616)]
#GetAtol
 # '1': [('p=8', {'base_estimator': 'RF', 'n_estimators': 300}, 0.8)], 
 # '2': [('p=8', {'base_estimator': 'RF', 'n_estimators': 200}, 0.6857142857142857)], 
 # '3': [('p=8', {'base_estimator': 'RF', 'n_estimators': 100}, 0.7476190476190477)], 
 # '4': [('p=4', {'base_estimator': 'RF', 'n_estimators': 100}, 0.8095238095238095)], 
 # '5': [('p=4', {'base_estimator': 'RF', 'n_estimators': 200}, 0.8666666666666668)], 
 # '6': [('p=8', {'C': 835.625671897373, 'base_estimator': 'SVM', 'gamma': 0.00018457575175565604, 'kernel': 'rbf'}, 0.8904761904761905)], 
 # '7': [('p=8', {'base_estimator': 'RF', 'n_estimators': 100}, 0.9238095238095239)], 
 # '8': [('p=8', {'base_estimator': 'RF', 'n_estimators': 100}, 0.8857142857142856)], 
 # '9': [('p=8', {'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'}, 0.8904761904761905)]}
#GetPersSilhouetteFeature
# '1': [('p=100  q=0',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.6238095238095238)], 
# '2': [('p=100  q=0',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5476190476190477)],
# '3': [('p=100  q=2',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5714285714285714),
#  ('p=200  q=2',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5714285714285714)],
# '4': [('p=100  q=2',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.7380952380952381)],
# '5': [('p=50  q=5',{'C': 879.1425034294132,'base_estimator': 'SVM','gamma': 0.0010352534930954075,'kernel': 'rbf'}, 0.8380952380952381),
#  ('p=100  q=5',{'C': 74.36417174259957,'base_estimator': 'SVM','gamma': 0.0063344256099350914,'kernel': 'rbf'},0.8380952380952381)],
# '6': [('p=50  q=2',{'C': 879.1425034294132,'base_estimator': 'SVM','gamma': 0.0010352534930954075,'kernel': 'rbf'},0.8857142857142858),
#  ('p=100  q=2',{'C': 74.36417174259957,'base_estimator': 'SVM','gamma': 0.0063344256099350914,'kernel': 'rbf'},0.8857142857142858)],
# '7': [('p=50  q=10',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.9),
#  ('p=100  q=10',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},  0.9),
#  ('p=200  q=10',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.9)],
# '8': [('p=100  q=1',{'C': 958.889530150502,'base_estimator': 'SVM','gamma': 0.007617800132393983, 'kernel': 'rbf'},0.8142857142857143)],
# '9': [('p=50  q=2',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.7190476190476192)],
# '10': [('p=100  q=0',{'base_estimator': 'RF', 'n_estimators': 100},0.5904761904761905)]
#GetComplexPolynomialFeature
# '1': [('p=10  q=S',{'base_estimator': 'RF', 'n_estimators': 200},0.4523809523809524)],
# '2': [('p=10  q=S',{'base_estimator': 'RF', 'n_estimators': 100},0.3238095238095238)],
# '3': [('p=20  q=S',{'base_estimator': 'RF', 'n_estimators': 200},0.40476190476190477)],
# '4': [('p=20  q=S',{'base_estimator': 'RF', 'n_estimators': 200},0.5571428571428572)],
# '5': [('p=20  q=S',{'base_estimator': 'RF', 'n_estimators': 300},0.7428571428571428)],
# '6': [('p=20  q=S',{'base_estimator': 'RF', 'n_estimators': 300},0.8666666666666666)],
# '7': [('p=10  q=S',{'base_estimator': 'RF', 'n_estimators': 50},0.880952380952381)],
# '8': [('p=20  q=T',{'base_estimator': 'RF', 'n_estimators': 50},0.8428571428571427)],
# '9': [('p=10  q=T',{'base_estimator': 'RF', 'n_estimators': 100},0.8047619047619048)],
# '10': [('p=20  q=T',{'base_estimator': 'RF', 'n_estimators': 200},0.8380952380952381)]}
# GetPersLandscapeFeature
# '1': [('p=200  q=5',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.6238095238095238)],
# '2': [('p=50  q=20',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5809523809523809),
#  ('p=100  q=20',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5809523809523809),
#  ('p=200  q=20',{'C': 1000.0405153241447,'base_estimator': 'SVM','degree': 2,'gamma': 0.0009688387165373345,'kernel': 'poly'},0.5809523809523809)],
# '3': [('p=200  q=20',{'C': 1000.0405153241447,'base_estimator': 'SVM','degree': 2,'gamma': 0.0009688387165373345,'kernel': 'poly'},0.6952380952380952)],
# '4': [('p=100  q=10',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},0.8238095238095238),
#       ('p=100  q=20',{'C': 998.1848109388686, 'base_estimator': 'SVM', 'kernel': 'linear'},   0.8238095238095238)], 
# '5': [('p=50  q=20',{'base_estimator': 'RF', 'n_estimators': 50},0.8904761904761905),
#  ('p=200  q=20',{'base_estimator': 'RF', 'n_estimators': 50},0.8904761904761905)],
# '6': [('p=50  q=20',{'base_estimator': 'RF', 'n_estimators': 50},0.9047619047619048)],
# '7': [('p=200  q=2',{'base_estimator': 'RF', 'n_estimators': 100},0.938095238095238)],
# '8': [('p=200  q=5',{'C': 212.62811600005904,'base_estimator': 'SVM','gamma': 0.0030862881073963535,'kernel': 'rbf'},0.8619047619047618)],
# '9': [('p=200  q=10',{'base_estimator': 'RF', 'n_estimators': 300},0.7761904761904763)],
# '10': [('p=50  q=10',{'C': 803.7575039373648, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5904761904761905),
#  ('p=50  q=20',{'C': 803.7575039373648, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5904761904761905),
#  ('p=100  q=10',{'C': 274.0499742167474, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5904761904761905),
#  ('p=100  q=20',{'C': 274.0499742167474, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5904761904761905),
#  ('p=200  q=10',{'C': 141.38693859523377, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5904761904761905),
#  ('p=200  q=20',{'C': 141.38693859523377, 'base_estimator': 'SVM', 'kernel': 'linear'},0.5904761904761905)]
#GetPersTropicalCoordinatesFeature
# '1': ({'base_estimator': 'RF', 'n_estimators': 200, 'r': 52.19579592311136},
#  0.5571428571428572),
# '2': ({'base_estimator': 'RF', 'n_estimators': 100, 'r': 19.234452089092336},
#  0.42857142857142855),
# '3': ({'base_estimator': 'RF', 'n_estimators': 100, 'r': 19.234452089092336},
#  0.6095238095238096),
# '4': ({'base_estimator': 'RF', 'n_estimators': 200, 'r': 52.19579592311136},
#  0.7666666666666666),
# '5': ({'base_estimator': 'RF', 'n_estimators': 200, 'r': 166.92513437947457},
#  0.8571428571428571),
# '6': ({'base_estimator': 'RF', 'n_estimators': 500, 'r': 104.95987537408911},
#  0.8761904761904763),
# '7': ({'base_estimator': 'RF', 'n_estimators': 50, 'r': 1.801599392208586},
#  0.8952380952380953),
# '8': ({'base_estimator': 'RF', 'n_estimators': 500, 'r': 165.82938147342833},
#  0.8428571428571429),
# '9': ({'base_estimator': 'RF', 'n_estimators': 50, 'r': 188.3673897506845},
#  0.7571428571428571),
# '10': ({'C': 429.0911898712949,'base_estimator': 'SVM','gamma': 0.033478475471326576,'kernel': 'rbf','r': 132.6882995636896},0.6666666666666667)}
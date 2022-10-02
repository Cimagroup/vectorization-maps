import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from OUTEX_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
             
def classification(func, str_p='', str_q='', base_estimator='RF', 
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3):
    
    if str_p!='':
        p = str_p
        str_p = '_'+str_p
    if str_q!='':
        str_q = '_'+str_q
            
    #perform the classification using the direct_optimization method for all
    #functions except for TropicalCoordinates, where tropical_optimization is
    #used instead
    
    path_feat = 'Outex-TC-00024/features/' 
    path_data = 'Outex-TC-00024/data/000/'
    

    train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                        usecols=[1]).to_numpy().flatten().tolist())
    test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                       usecols=[1]).to_numpy().flatten().tolist())
    labels = np.hstack([train_labels, test_labels])
    Z_train, Z_test, y_train, y_test = train_test_split(range(2720), labels, 
                                                        test_size=0.3, 
                                                        random_state=0)
    
    if func!=GetPersTropicalCoordinatesFeature:
        with open(path_feat + func.__name__ + '_l_d0.pkl', 'rb') as f:
            features_l_d0 = pickle.load(f)
        with open(path_feat + func.__name__ + '_l_d1.pkl', 'rb') as f:
            features_l_d1 = pickle.load(f)
        with open(path_feat + func.__name__ + '_u_d0.pkl', 'rb') as f:
            features_u_d0 = pickle.load(f)
        with open(path_feat + func.__name__ + '_u_d1.pkl', 'rb') as f:
            features_u_d1 = pickle.load(f)
        
    
        
        method = main_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree)
        
        X_train = []
        for i in Z_train:
            X_train.append(
                np.hstack(
                    [
                        features_l_d0[str(i) + str_p + str_q],
                        features_l_d1[str(i) + str_p + str_q],
                        features_u_d0[str(i) + str_p + str_q],
                        features_u_d1[str(i) + str_p + str_q]
                    ]
                    ))
            
        X_test = []
        for i in Z_test:
            X_test.append(
                np.hstack(
                    [
                        features_l_d0[str(i) + str_p + str_q],
                        features_l_d1[str(i) + str_p + str_q],
                        features_u_d0[str(i) + str_p + str_q],
                        features_u_d1[str(i) + str_p + str_q]
                    ]
                    ))    
        X_train = float64to32(X_train)
        X_test = float64to32(X_test)
    
    else:
        method = tropical_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree, r=float(p))
        
        X_train, X_test = Z_train, Z_test
    
    score_list = []
    for i in range(100):
        method.fit(X_train, y_train)
        score_list.append(np.mean(y_test.ravel() == method.predict(X_test)))
        
    print(np.mean(score_list))
    
# classification(func = GetPersStats, base_estimator='RF', n_estimators=300)
# 0.8964460784313725
# classification(func = GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=200)
# 0.84453431372549
# classification(str_q = '100', func = GetPersEntropyFeature, base_estimator='RF', n_estimators=300)
# 0.8197916666666666
# classification(str_p = '100', func = GetBettiCurveFeature,  base_estimator='SVM', kernel='poly', C=1000.0405153241447 , gamma=0.0009688387165373345, degree=2)
# 0.8186274509803918
# classification(str_p = '100', func = GetPersLifespanFeature, base_estimator='RF', n_estimators=300)
# 0.8450367647058823
# classification(func=GetAtolFeature, str_p='64', C=998.1848109388686, base_estimator='SVM', kernel='linear')
# 0.8370098039215691
# classification(str_p='150', func=GetPersImageFeature, base_estimator='RF', n_estimators=300)
# 0.7597794117647058
# classification(func=GetPersSilhouetteFeature, str_p='100', str_q='1', base_estimator='SVM', kernel='poly', degree=2, C=1000.0405153241447, gamma=0.0009688387165373345)
# 0.8370098039215691
# classification(func=GetPersLandscapeFeature, str_p='50', str_q='20', C=998.1848109388686, base_estimator='SVM', kernel='linear')
# 0.8370098039215691
# classification(func=GetPersTropicalCoordinatesFeature, str_p='492.5731592803383', base_estimator='RF', n_estimators=200)
# 0.872732843137255

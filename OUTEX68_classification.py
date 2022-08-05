import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from OUTEX68_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC

#classification(func=GetPersStatsFeature, base_estimator='RF', n_estimators=500)
#classification(func=GetCarlssonCoordinatesFeature , base_estimator='RF', n_estimators=500)
#classification(func=GetPersEntropyFeature, str_p='200', base_estimator='RF', n_estimators=300)
#classification(func=GetBettiCurveFeature, str_p='200', base_estimator='SVM', C=105, kernel='poly', gamma=0.00891, degree=2)
#classification(func=GetPersLifeSpanFeature, str_p='100', base_estimator='SVM', C=211.4, kernel='linear')
#classification(func=GetPersImageFeature, str_p='225', , base_estimator='RF', n_estimators=300)
#classification(func=GetAtolFeature, str_p='2', base_estimator='RF', n_estimators=500)
#classification(func=GetPersSilhouetteFeature, str_p='100', str_q='5', base_estimator='RF', n_estimators=300)
#classification(func=GetComplexPolynomialFeature, str_p='5', str_q='R', base_estimator='RF', n_estimators=300)
#classification(func=GetPersLandscapeFeature, str_p='50', str_q='20', base_estimator='SVM', C=70, kernel='linear')
#classification(func=GetPersTropicalCoordinatesFeature, str_p='258.53', base_estimator='RF', n_estimators=300)
             
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
    
    else:
        method = tropical_classifier(base_estimator, 
                     n_estimators, C, kernel, gamma, degree, r=float(p))
        
        X_train, X_test = Z_train, Z_test
    
    method.fit(X_train, y_train) 
    y_pred = method.predict(X_test)
    print(classification_report(y_test, y_pred))
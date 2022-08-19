import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from OUTEX_tropical_optimisation import tropical_classifier
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
             
def classification(func, str_p='', str_q='', base_estimator='RF', t=10,
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3):
    
    data_path = 'Shrec14/data/'
    data = pd.read_csv(data_path+'Uli_data.csv')
    
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
    
    labelsD = labels[0].tolist()
    label_names = label_names[0].tolist()
    
    if func!=GetPersTropicalCoordinatesFeature:
        X = []
        dgms = dgmsDF[dgmsDF.trial==t]
        dgms = np.array(dgms['Dgm1'])
        for i in range(dgms.shape[0]):
            X.append(func(dgms[i]))          
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            labelsD, 
                                                            test_size=0.3, 
                                                            random_state=t)
        
    
        method = main_classifier(base_estimator,
                     n_estimators, C, kernel, gamma, degree)
        
    
    method.fit(X_train, y_train) 
    y_pred = method.predict(X_test)
    print(classification_report(y_test, y_pred))


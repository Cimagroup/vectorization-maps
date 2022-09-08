import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from OUTEX_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
from numpy.random import seed
import copy

             
def classification(func, str_p='', str_q='', base_estimator='RF', t=6,
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3):
    
    data_path = 'Shrec14/data/'
    data = pd.read_csv(data_path+'Uli_data.csv')
    seed(1)
    
    if str_p!='':
        p = int(str_p)
        str_p = '_'+str_p
    if str_q!='':
        if str.isdigit(str_q):
            q = int(str_q)
        else:
            q = copy.deepcopy(str_q)
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
            if str_q!='':
                X.append(func(dgms[i],p,q))
            elif str_p!='':
                X.append(func(dgms[i],p))
            else:
                X.append(func(dgms[i]))
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            labelsD, 
                                                            test_size=0.3, 
                                                            random_state=t)
    
        method = main_classifier(base_estimator,
                     n_estimators, C, kernel, gamma, degree)
    
    score_list = []
    for i in range(100):
        method.fit(X_train, y_train)
        score_list.append(np.mean(y_test == method.predict(X_test)))
        
    print(np.mean(score_list))

#GetPersStats
# classification(t=1, func=GetPersStats, base_estimator='SVM', kernel='linear', C=172.00627184565752)
# classification(t=2, func=GetPersStats, base_estimator='SVM', kernel='linear', C=857.308148101551)
# classification(t=3, func=GetPersStats, base_estimator='RF', n_estimators=500)
# classification(t=4, func=GetPersStats, base_estimator='SVM', kernel='rbf', C=520.1575753881025, gamma=0.0004298166684598349)
# classification(t=5, func=GetPersStats, base_estimator='SVM', kernel='poly', C=481.94539983958083, gamma=0.004709299229425083, degree=2)
# classification(t=6, func=GetPersStats, base_estimator='SVM', kernel='rbf', C=99.69875067655931, gamma=0.007178703553818194)
# classification(t=7, func=GetPersStats, base_estimator='SVM', kernel='rbf', C=666.4119991108059, gamma=0.002793907374186834)
# classification(t=8, func=GetPersStats, base_estimator='SVM', kernel='linear', C=208.3384891235931)
# classification(t=9, func=GetPersStats, base_estimator='RF', n_estimators=100)
# classification(t=10, func=GetPersStats, base_estimator='RF', n_estimators=300)
#GetCarlssonCoordinatesFeature
# classification(t=1, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=50)
# classification(t=2, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=50)
# classification(t=3, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=100)
# classification(t=4, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=200)
# classification(t=5, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=200)
# classification(t=6, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=50)
# classification(t=7, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=50)
# classification(t=8, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=300)
# classification(t=9, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=500)
# classification(t=10, func=GetCarlssonCoordinatesFeature, base_estimator='RF', n_estimators=50)
#GetPersEntropyFeature
# classification(t=1, str_p='100', base_estimator='SVM', kernel='rbf', C=922.9301058561884, gamma=0.007380093642715903, func=GetPersEntropyFeature)
# classification(t=2, str_p='100', base_estimator='SVM', kernel='rbf', C=216.79764711920058, gamma=0.007686298833109734, func=GetPersEntropyFeature)
# classification(t=3, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=500, func=GetPersEntropyFeature)
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=200, func=GetPersEntropyFeature)
# classification(t=6, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersEntropyFeature)
# classification(t=7, str_p='200', base_estimator='RF', n_estimators=200, func=GetPersEntropyFeature)
# classification(t=8, str_p='200', base_estimator='SVM', kernel='rbf', C=987.1076998901724, gamma=0.0021546383452133513, func=GetPersEntropyFeature)
# classification(t=9, str_p='200', base_estimator='SVM', kernel='rbf', C=299.95014695881906, gamma=0.02142516136865941, func=GetPersEntropyFeature)
# classification(t=10, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersEntropyFeature) 
# GetBettiCurveFeature
# classification(t=1, str_p='200', base_estimator='SVM', kernel='rbf',C=380.93803081127777,gamma=0.001874756626007183, func=GetBettiCurveFeature)
# classification(t=2, str_p='200', base_estimator='SVM', kernel='rbf',C=306.7083011841142,gamma=0.0010742455207731478, func=GetBettiCurveFeature)
# classification(t=3, str_p='200', base_estimator='SVM', kernel='rbf',C=430.2234989885916,gamma=0.0014191616950970255, func=GetBettiCurveFeature)
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=100, func=GetBettiCurveFeature)
# classification(t=5, str_p='200', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# classification(t=6, str_p='100', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# classification(t=7, str_p='200', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# classification(t=8, str_p='200', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# classification(t=9, str_p='100', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# classification(t=10, str_p='50', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
#GetPersLifespanCurve
# classification(t=1, str_p='200', base_estimator='SVM', kernel='linear',C=407.3866802625573, func=GetPersLifespanFeature)
# classification(t=2, str_p='200', base_estimator='SVM', kernel='poly', C=236.8405359150676, gamma=0.003181360558229595, degree=2, func=GetPersLifespanFeature)
# classification(t=3, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersLifespanFeature)
# classification(t=4, str_p='100', base_estimator='RF', n_estimators=100, func=GetPersLifespanFeature)
# classification(t=5, str_p='200', base_estimator='RF', n_estimators=500, func=GetPersLifespanFeature)
# classification(t=6, str_p='200', base_estimator='RF', n_estimators=200, func=GetPersLifespanFeature)
# classification(t=7, str_p='100', base_estimator='RF', n_estimators=50, func=GetPersLifespanFeature)
# classification(t=8, str_p='100', base_estimator='SVM', kernel='rbf',C=587.0275746378663, gamma=0.0017317618758819116, func=GetPersLifespanFeature)
# classification(t=9, str_p='200', base_estimator='SVM', kernel='rbf',C=847.1285551888576, gamma=0.012666621559834168, func=GetPersLifespanFeature)
# classification(t=10, str_p='100', base_estimator='RF', n_estimators=300, func=GetPersLifespanFeature)
#GetPersImageFeature
# classification(t=1, str_p='200', base_estimator='RF', n_estimators=500, func=GetPersImageFeature)
# classification(t=2, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersImageFeature)
# classification(t=3, str_p='100', base_estimator='RF', n_estimators=300, func=GetPersImageFeature)
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersImageFeature)
# classification(t=5, str_p='50', base_estimator='RF', n_estimators=50, func=GetPersImageFeature)
# classification(t=6, str_p='10', base_estimator='RF', n_estimators=300, func=GetPersImageFeature)
# classification(t=7, str_p='50', base_estimator='RF', n_estimators=300, func=GetPersImageFeature)
# classification(t=8, str_p='25', base_estimator='SVM', kernel='rbf',C=737.8784171383774,gamma=0.006735651377937472, func=GetPersImageFeature)
# classification(t=9, str_p='25', base_estimator='RF', n_estimators=200, func=GetPersImageFeature)
# classification(t=10, str_p='50', base_estimator='RF', n_estimators=100, func=GetPersImageFeature)



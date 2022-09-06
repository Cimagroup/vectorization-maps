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


             
def classification(func, str_p='', str_q='', base_estimator='RF', t=10,
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3):
    
    data_path = 'Shrec14/data/'
    data = pd.read_csv(data_path+'Uli_data.csv')
    seed(1)
    
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
    
    score_list = []
    for i in range(100):
        method.fit(X_train, y_train)
        score_list.append(np.mean(y_test == method.predict(X_test)))
        
    print(np.mean(score_list))

#GetPersStats
# classification(func=GetPersStats, base_estimator='SVM', t=1, C=242.61602391361458, kernel='linear')
# classification(func=GetPersStats, base_estimator='SVM', t=2, C=34.41327954736772, kernel='linear')
# classification(func=GetPersStats, base_estimator='SVM', t=3, C=327.14773416776956, kernel='linear')
# classification(func=GetPersStats, base_estimator='SVM', t=4, C=75.87960668267124, kernel='linear')
# classification(func=GetPersStats, base_estimator='SVM', t=5, C=6.063748502539767, kernel='linear')
# classification(func=GetPersStats, base_estimator='SVM', t=6, C=547.0104892185311, kernel='rbf', gamma=0.0022931176685105576)
# classification(func=GetPersStats, base_estimator='SVM', t=7, C=277.311870239477, kernel='rbf', gamma=0.005724516397732078)
# classification(func=GetPersStats, base_estimator='SVM', t=8, C=771.0945390318818, kernel='poly', gamma=0.0180808489464352, degree=2)
# classification(func=GetPersStats, base_estimator='RF', t=9, n_estimators=300)
# classification(func=GetPersStats, base_estimator='SVM', t=10, C=219.44680984683285, kernel='linear')
#GetCarlssonCoordinatesFeature
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=1, n_estimators=200)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=2, n_estimators=50)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=3, n_estimators=50)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=4, n_estimators=100)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=5, n_estimators=300)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=6, n_estimators=100)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=7, n_estimators=100)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=8, n_estimators=50)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=9, n_estimators=300)
# classification(func=GetCarlssonCoordinatesFeature, base_estimator='RF', t=10, n_estimators=100)
#GetPersEntropyFeature
# classification(func=GetPersEntropyFeature, str_p='200', t=1, base_estimator='SVM', kernel='rbf', C=333.73240112817837, gamma=0.004816329290999086)
# classification(func=GetPersEntropyFeature, str_p='200', t=2, base_estimator='RF', n_estimators=50)
# classification(func=GetPersEntropyFeature, str_p='200', t=3, base_estimator='RF', n_estimators=500)
# classification(func=GetPersEntropyFeature, str_p='200', t=4, base_estimator='RF', n_estimators=300)
# classification(func=GetPersEntropyFeature, str_p='200', t=5, base_estimator='RF', n_estimators=300)
# classification(func=GetPersEntropyFeature, str_p='200', t=6, base_estimator='RF', n_estimators=300)
# classification(func=GetPersEntropyFeature, str_p='200', t=7, base_estimator='RF', n_estimators=300)
# classification(func=GetPersEntropyFeature, str_p='200', t=8, base_estimator='SVM', kernel='rbf', C=137.5518127520976, gamma=0.001966757624327403)
# classification(func=GetPersEntropyFeature, str_p='100', t=9, base_estimator='SVM', kernel='rbf', C=167.17443031566526, gamma=0.0029046626574237167)
# classification(func=GetPersEntropyFeature, str_p='100', t=10, base_estimator='SVM', kernel='rbf', C=236.75530013409286, gamma=0.004904869721469778)
#GetBettiCurveFeature
# classification(func=GetBettiCurveFeature, str_p='200', t=1, base_estimator='SVM', kernel='poly', C=685.1757925863737, gamma=0.026095092746128038, degree=3)
# classification(func=GetBettiCurveFeature, str_p='100', t=2, base_estimator='SVM', kernel='poly', C=707.6263512818409, gamma=0.010457978529920224, degree=2)
# classification(func=GetBettiCurveFeature, str_p='100', t=3, base_estimator='SVM', kernel='rbf', C=206.01770354273862, gamma=0.0030882237830110715)
# classification(func=GetBettiCurveFeature, str_p='200', t=4, base_estimator='RF', n_estimators=50)
# classification(func=GetBettiCurveFeature, str_p='200', t=5, base_estimator='RF', n_estimators=300)
# classification(func=GetBettiCurveFeature, str_p='200', t=6, base_estimator='RF', n_estimators=300)
# classification(func=GetBettiCurveFeature, str_p='200', t=7, base_estimator='RF', n_estimators=50)
# classification(func=GetBettiCurveFeature, str_p='200', t=8, base_estimator='RF', n_estimators=100)
# classification(func=GetBettiCurveFeature, str_p='200', t=9, base_estimator='SVM', kernel='rbf', C=45.91396006997594, gamma=0.0002056231610313888)
# classification(func=GetBettiCurveFeature, str_p='200', t=10, base_estimator='SVM', kernel='poly', C=838.3106508517023, gamma=0.03928745645869667, degree=2)
#GetPersLifespanFeature
# classification(func=GetPersLifespanFeature, str_p='200', t=1, base_estimator='SVM', kernel='poly', C=879.5184108555763, gamma=0.0349498757941561, degree=2)
# classification(func=GetPersLifespanFeature, str_p='200', t=2, base_estimator='SVM', kernel='linear',C=775.2915571850108)
# classification(func=GetPersLifespanFeature, str_p='200', t=3, base_estimator='SVM', kernel='linear',C=205.51602581712015)
# classification(func=GetPersLifespanFeature, str_p='100', t=4, base_estimator='RF',n_estimators=300)
# classification(func=GetPersLifespanFeature, str_p='200', t=5, base_estimator='RF',n_estimators=300)
# classification(func=GetPersLifespanFeature, str_p='100', t=6, base_estimator='RF',n_estimators=200)
# classification(func=GetPersLifespanFeature, str_p='200', t=7, base_estimator='RF',n_estimators=300)
# classification(func=GetPersLifespanFeature, str_p='200', t=8, base_estimator='SVM', kernel='linear',C=809.5469892495846)
# classification(func=GetPersLifespanFeature, str_p='200', t=9, base_estimator='SVM', kernel='linear',C=142.0011634426691)
# classification(func=GetPersLifespanFeature, str_p='200', t=10, base_estimator='RF', n_estimators=300)
#GetPersImage
# classification(func=GetPersImageFeature, str_p='200', t=1, base_estimator='RF',n_estimators=500)
# classification(func=GetPersImageFeature, str_p='200', t=2, base_estimator='RF',n_estimators=300)
# classification(func=GetPersImageFeature, str_p='200', t=3, base_estimator='RF',n_estimators=500)
# classification(func=GetPersImageFeature, str_p='25', t=4, base_estimator='RF',n_estimators=200)
# classification(func=GetPersImageFeature, str_p='50', t=5, base_estimator='RF',n_estimators=50)
# classification(func=GetPersImageFeature, str_p='10', t=6, base_estimator='RF',n_estimators=300)
# classification(func=GetPersImageFeature, str_p='100', t=7, base_estimator='SVM',kernel='rbf',C=108.15265139116536, gamma=0.0007232398939433436)
# classification(func=GetPersImageFeature, str_p='10', t=8, base_estimator='SVM',kernel='linear',C=360.00349917907647)
# classification(func=GetPersImageFeature, str_p='25', t=9, base_estimator='RF',n_estimators=200)
# classification(func=GetPersImageFeature, str_p='25', t=10, base_estimator='RF',n_estimators=50)
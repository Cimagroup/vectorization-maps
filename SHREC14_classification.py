import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from direct_optimisation import main_classifier
from SHREC14_tropical_optimisation import tropical_classifier
from vectorisation import *
from sklearn.svm import SVC
from numpy.random import seed
import copy

             
def classification(func, str_p='', str_q='', base_estimator='RF', t=6,
             n_estimators=100, C=1.0, kernel='rbf', gamma=0.1, degree=3):
    
    feat_path = "Shrec14/features/"
    seed(1)
    
    #extract the parameters of the method
    if str_p!='':
        if np.char.isdigit(str_p):
            p = int(str_p)
        elif np.char.isdigit(str_p.replace('.','',1)):
            p = float(str_p)
        else:
            p = copy.deepcopy(str_p)
            str_p = '_'+str_p
    if str_q!='':
        if np.char.isdigit(str_q):
            q = int(str_q)
        elif np.char.isdigit(str_q.replace('.','',1)):
            q = float(str_q)
        else:
            q = copy.deepcopy(str_q)
            str_q = '_'+str_q
        
    #Load the labels
    with open(feat_path + 'Z_train' +'.pkl', 'rb') as f:
        Z_train = pickle.load(f)
    with open(feat_path + 'Z_test' +'.pkl', 'rb') as f:
        Z_test = pickle.load(f)
    with open(feat_path + 'y_train' +'.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(feat_path + 'y_test' +'.pkl', 'rb') as f:
        y_test = pickle.load(f)
        

    
    if func!=GetPersTropicalCoordinatesFeature:
        with open(feat_path + func.__name__ +'.pkl', 'rb') as f:
            features = pickle.load(f)
        X_train = []
        for i in Z_train[str(t)]:
            if str_q!='':
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
            elif str_p!='':
                X_train.append(features[str(t)+'_'+str(p)+'_'+str(i)])
            else:
                X_train.append(features[str(t)+'_'+str(i)])
        X_test = []
        for i in Z_test[str(t)]:
            if str_q!='':
                X_test.append(features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)])
            elif str_p!='':
                X_test.append(features[str(t)+'_'+str(p)+'_'+str(i)])
            else:
                X_test.append(features[str(t)+'_'+str(i)])
    
        method = main_classifier(base_estimator=base_estimator,
                     n_estimators=n_estimators, C=C, kernel=kernel, 
                     gamma=gamma, degree=degree)
    else:
        with open(feat_path + 'dgmsT' +'.pkl', 'rb') as f:
            dgmsT = pickle.load(f)
        method = tropical_classifier(base_estimator=base_estimator,  t=t, 
                     n_estimators=n_estimators, C=C, gamma=gamma, r=p, 
                     degree=degree, kernel=kernel,  dgmsT=dgmsT)
        X_train=Z_train[str(t)]
        X_test=Z_test[str(t)]
    score_list = []
    for i in range(10):
        method.fit(X_train, y_train[str(t)])
        score_list.append(np.mean(y_test[str(t)] == method.predict(X_test)))
        
    print(np.mean(score_list))

# GetPersStats
# classification(t=1,base_estimator='RF', n_estimators=50, func=GetPersStats)
# 0.581111111111111
# classification(t=2,base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345, degree=2, func=GetPersStats)
# 0.6777777777777778
# classification(t=3,base_estimator='SVM', kernel='linear', C=141.38693859523377, func=GetPersStats)
# 0.6555555555555557
# classification(t=4,base_estimator='SVM', kernel='linear', C=141.38693859523377, func=GetPersStats)
# 0.5888888888888888
# classification(t=5,base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersStats)
# 0.9
# classification(t=6,base_estimator='SVM', kernel='rbf', C=288.77533858634877, gamma=0.001392949093880637,func=GetPersStats)
# 0.8555555555555555
# classification(t=7,base_estimator='SVM', kernel='rbf', C=936.5390708060319, gamma=0.01872823656893796,func=GetPersStats)
# 0.9222222222222223
# classification(t=8,base_estimator='SVM', kernel='linear', C=803.7575039373648,func=GetPersStats)
# 0.8777777777777779
# classification(t=9, base_estimator='SVM', kernel='linear', C=686.2195003967595,func=GetPersStats)
# 0.8555555555555555
# classification(t=10,base_estimator='RF', n_estimators=100, func=GetPersStats)
# 0.8455555555555556

# GetCarlssonCoordinates
# classification(t=1,base_estimator='RF',n_estimators=50,func=GetCarlssonCoordinatesFeature)
# 0.4
# classification(t=2,base_estimator='RF',n_estimators=200,func=GetCarlssonCoordinatesFeature)
# 0.33111111111111113
# classification(t=3,base_estimator='RF',n_estimators=200,func=GetCarlssonCoordinatesFeature)
# 0.37
# classification(t=4,base_estimator='RF',n_estimators=100,func=GetCarlssonCoordinatesFeature)
# 0.571111111111111
# classification(t=5,base_estimator='RF',n_estimators=300,func=GetCarlssonCoordinatesFeature)
# 0.9066666666666668
# classification(t=6,base_estimator='RF',n_estimators=50,func=GetCarlssonCoordinatesFeature)
# 0.8611111111111113
# classification(t=7,base_estimator='RF',n_estimators=50,func=GetCarlssonCoordinatesFeature)
# 0.8555555555555555
# classification(t=9,base_estimator='RF',n_estimators=100,func=GetCarlssonCoordinatesFeature)
# 0.7422222222222222
# classification(t=9,base_estimator='RF',n_estimators=100,func=GetCarlssonCoordinatesFeature)
# 0.7422222222222222
# classification(t=10,base_estimator='RF',n_estimators=200,func=GetCarlssonCoordinatesFeature)
# 0.678888888888889

# GetPersEntropyFeature
# classification(t=1, str_p='200', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersEntropyFeature)
# 0.47777777777777775
# classification(t=2, str_p='100', base_estimator='SVM', kernel='poly', C=223.1245475353748, degree=3, gamma=0.005940143639429267, func=GetPersEntropyFeature)
# 0.3222222222222223
# classification(t=3, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersEntropyFeature)
# 0.4644444444444445
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.5344444444444443
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.6466666666666667
# classification(t=6, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.7155555555555555
# classification(t=7, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.6644444444444446
# classification(t=8, str_p='200', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersEntropyFeature)
# 0.5777777777777777
# classification(t=9, str_p='50', base_estimator='SVM', kernel='rbf', C=879.1425034294132, gamma=0.0010352534930954075, func=GetPersEntropyFeature)
# 0.4666666666666667
# classification(t=10, str_p='200', n_estimators=500, func=GetPersEntropyFeature)
# 0.39333333333333337

# GetBettiCurveFeature
# classification(t=1, str_p='200', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetBettiCurveFeature)
# 0.5444444444444444
# classification(t=2, str_p='200', base_estimator='SVM', kernel='rbf', C=288.77533858634877, gamma=0.001392949093880637, func=GetBettiCurveFeature)
# 0.5444444444444444
# classification(t=3, str_p='200', base_estimator='SVM', kernel='rbf', C=288.77533858634877, gamma=0.001392949093880637, func=GetBettiCurveFeature)
# 0.7111111111111111
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# 0.6577777777777778
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# 0.6777777777777778
# classification(t=6, str_p='200', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# 0.6322222222222221
# classification(t=7, str_p='200', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# 0.6633333333333334
# classification(t=8, str_p='200', base_estimator='RF', n_estimators=50, func=GetBettiCurveFeature)
# 0.6322222222222221
# classification(t=9, str_p='100', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# 0.5533333333333333
# classification(t=10, str_p='50', base_estimator='RF', n_estimators=50, func=GetBettiCurveFeature)
# 0.34

# GetPersLifespanFeature
# classification(t=1, str_p='200', base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345, degree=2, func=GetPersLifespanFeature)
# 0.47777777777777775
# classification(t=2, str_p='100', base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345, degree=2, func=GetPersLifespanFeature)
# 0.4
# classification(t=3, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersLifespanFeature)
# 0.5844444444444443
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=50, func=GetPersLifespanFeature)
# 0.64
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=300, func=GetPersLifespanFeature)
# 0.7611111111111111
# classification(t=6, str_p='100', base_estimator='RF', n_estimators=500, func=GetPersLifespanFeature)
# 0.8533333333333333
# classification(t=7, str_p='200', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetPersLifespanFeature)
# 0.8666666666666668
# classification(t=8, str_p='50', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetPersLifespanFeature)
# 0.8222222222222222
# classification(t=9, str_p='50', base_estimator='SVM', kernel='rbf', C=141.38693859523377, func=GetPersLifespanFeature)
# 0.5888888888888888
# classification(t=10, str_p='200', base_estimator='RF', n_estimators=500, func=GetPersLifespanFeature)
# 0.5455555555555556

#GetPersImageFeature
# classification(t=1, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersImageFeature)
# 0.7055555555555555
# classification(t=2, str_p='100', base_estimator='RF', n_estimators=100, func=GetPersImageFeature)
# 0.7
# classification(t=3, str_p='100', base_estimator='RF', n_estimators=500, func=GetPersImageFeature)
# 0.7911111111111111
# classification(t=4, str_p='100', base_estimator='RF', n_estimators=500, func=GetPersImageFeature)
# 0.821111111111111
# classification(t=5, str_p='50', base_estimator='RF', n_estimators=50, func=GetPersImageFeature)
# 0.9066666666666666
# classification(t=6, str_p='10', base_estimator='RF', n_estimators=200, func=GetPersImageFeature)
# 0.7877777777777778
# classification(t=7, str_p='200', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetPersImageFeature)
# 0.8666666666666668
# classification(t=8, str_p='25', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetPersImageFeature)
# 0.7888888888888889
# classification(t=9, str_p='25', base_estimator='RF', n_estimators=100, func=GetPersImageFeature)
# 0.611111111111111
# classification(t=10, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersImageFeature)
# 0.4322222222222223

#GetTopologicalVectorFeature
# classification(t=1, str_p='20', n_estimators=100, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.3
# classification(t=2, str_p='20', n_estimators=50, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.21444444444444444
# classification(t=3, str_p='20', n_estimators=50, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.2677777777777778
# classification(t=4, str_p='5', n_estimators=50, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.1622222222222222
# classification(t=5, str_p='20', n_estimators=500, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.5477777777777778
# classification(t=6, str_p='20', n_estimators=200, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.6600000000000001
# classification(t=7, str_p='20', n_estimators=50, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.6477777777777778
# classification(t=8, str_p='20', n_estimators=300, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.7233333333333334
# classification(t=9, str_p='20', n_estimators=100, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.6122222222222222
# classification(t=10, str_p='20', n_estimators=200, base_estimator='RF', func=GetTopologicalVectorFeature)
# 0.44333333333333336

#GetPerSilhouetteFeature
# classification(t=1, str_p='100', str_q='0', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersSilhouetteFeature)
# 0.4111111111111111
# classification(t=2, str_p='100', str_q='0', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersSilhouetteFeature)
# 0.5333333333333333
# classification(t=3, str_p='100', str_q='2', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersSilhouetteFeature)
# 0.5222222222222223
# classification(t=4, str_p='100', str_q='2', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersSilhouetteFeature)
# 0.5444444444444444
# classification(t=5, str_p='50', str_q='5', base_estimator='SVM', kernel='rbf', C=879.1425034294132, gamma=0.0010352534930954075, func=GetPersSilhouetteFeature)
# 0.8444444444444444
# classification(t=6, str_p='50', str_q='2', base_estimator='SVM', kernel='rbf', C=879.1425034294132, gamma=0.0010352534930954075, func=GetPersSilhouetteFeature)
# 0.8
# classification(t=7, str_p='50', str_q='10', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersSilhouetteFeature)
# 0.8555555555555555
# classification(t=8, str_p='100', str_q='1', base_estimator='SVM', kernel='rbf', C=958.889530150502, gamma=0.007617800132393983, func=GetPersSilhouetteFeature)
# 0.8444444444444444
# classification(t=9, str_p='50', str_q='2', base_estimator='SVM', kernel='linear', C=998.1848109388686,  func=GetPersSilhouetteFeature)
# 0.7111111111111111
# classification(t=10, str_p='100', str_q='0', base_estimator='RF', n_estimators=100, func=GetPersSilhouetteFeature)
# 0.5411111111111111

#GetComplexPolynomialFeature
# classification(t=1, str_p='10', str_q='S', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.3188888888888889
# classification(t=2, str_p='10', str_q='S', base_estimator='RF', n_estimators=100, func=GetComplexPolynomialFeature)
# 0.34
# classification(t=3, str_p='20', str_q='S', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.37222222222222223
# classification(t=4, str_p='20', str_q='S', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.45111111111111113
# classification(t=5, str_p='20', str_q='S', base_estimator='RF', n_estimators=300, func=GetComplexPolynomialFeature)
# 0.7622222222222221
# classification(t=6, str_p='20', str_q='S', base_estimator='RF', n_estimators=300, func=GetComplexPolynomialFeature)
# 0.8255555555555555
# classification(t=7, str_p='10', str_q='S', base_estimator='RF', n_estimators=50, func=GetComplexPolynomialFeature)
# 0.8255555555555555
# classification(t=8, str_p='20', str_q='T', base_estimator='RF', n_estimators=100, func=GetComplexPolynomialFeature)
# 0.8544444444444445
# classification(t=9, str_p='10', str_q='T', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.7755555555555554
# classification(t=10, str_p='20', str_q='T', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.7066666666666667

#GetPersLandscapeFeature
# classification(t=1, str_p='200', str_q='5', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersLandscapeFeature)
# 0.5666666666666667
# classification(t=2, str_p='50', str_q='20', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersLandscapeFeature)
# 0.6222222222222221
# classification(t=3, str_p='200', str_q='20', base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345,  degree=2, func=GetPersLandscapeFeature)
# 0.7222222222222222
# classification(t=4, str_p='100', str_q='10', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersLandscapeFeature)
# 0.7222222222222222
# classification(t=5, str_p='200', str_q='20', base_estimator='RF', n_estimators=50, func=GetPersLandscapeFeature)
# 0.8877777777777778
# classification(t=6, str_p='50', str_q='20', base_estimator='RF', n_estimators=50, func=GetPersLandscapeFeature)
# 0.8666666666666668
# classification(t=7, str_p='200', str_q='2', base_estimator='RF', n_estimators=100, func=GetPersLandscapeFeature)
# 0.8466666666666667
# classification(t=8, str_p='200', str_q='5', base_estimator='SVM', kernel='rbf', C=212.62811600005904, gamma=0.0030862881073963535, func=GetPersLandscapeFeature)
# 0.8555555555555555
# classification(t=9, str_p='200', str_q='10', base_estimator='RF', n_estimators=300, func=GetPersLandscapeFeature)
# 0.8144444444444444
# classification(t=10, str_p='100', str_q='10', base_estimator='SVM', kernel='linear', C=274.0499742167474, func=GetPersLandscapeFeature)
# 0.5

#GetPersTropicalCoordinatesFeature
# classification(t=1, base_estimator='RF', n_estimators=200, str_p='52.19579592311136', func=GetPersTropicalCoordinatesFeature)
# classification(t=2, base_estimator='RF', n_estimators=100, str_p='19.234452089092336', func=GetPersTropicalCoordinatesFeature)
# 0.4322222222222223
# classification(t=3, base_estimator='RF', n_estimators=100, str_p='19.234452089092336', func=GetPersTropicalCoordinatesFeature)
# 0.4922222222222222
# classification(t=4, base_estimator='RF', n_estimators=200, str_p='52.19579592311136', func=GetPersTropicalCoordinatesFeature)
# 0.6066666666666667
# classification(t=5, base_estimator='RF', n_estimators=200, str_p='166.92513437947457', func=GetPersTropicalCoordinatesFeature)
# 0.8700000000000001
# classification(t=6, base_estimator='RF', n_estimators=500, str_p='104.95987537408911', func=GetPersTropicalCoordinatesFeature)
# 0.8622222222222222
# classification(t=7, base_estimator='RF', n_estimators=50, str_p='1.801599392208586', func=GetPersTropicalCoordinatesFeature)
# 0.8322222222222221
# classification(t=8, base_estimator='RF', n_estimators=500, str_p='165.82938147342833', func=GetPersTropicalCoordinatesFeature)
# 0.7777777777777777
# classification(t=9, base_estimator='RF', n_estimators=50, str_p='188.3673897506845', func=GetPersTropicalCoordinatesFeature)
# 0.7411111111111112
# classification(t=10, base_estimator='SVM', C=429.0911898712949, kernel='rbf', gamma=0.033478475471326576, str_p='132.6882995636896', func=GetPersTropicalCoordinatesFeature)
# 0.5999999999999999



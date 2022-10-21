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
    
    for i in range(100):
        method.fit(X_train, y_train[str(t)])
        score_list.append(np.mean(y_test[str(t)] == method.predict(X_test)))
        
    print(np.mean(score_list))

# print('GetPersStats')
# classification(t=1,base_estimator='RF', n_estimators=50, func=GetPersStats)
# 0.5752222222222221
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
# 0.8441111111111113

# print('GetCarlssonCoordinates')
# classification(t=1,base_estimator='RF',n_estimators=50,func=GetCarlssonCoordinatesFeature)
# 0.4077777777777778
# classification(t=2,base_estimator='RF',n_estimators=200,func=GetCarlssonCoordinatesFeature)
# 0.3363333333333334
# classification(t=3,base_estimator='RF',n_estimators=200,func=GetCarlssonCoordinatesFeature)
# 0.37755555555555553
# classification(t=4,base_estimator='RF',n_estimators=100,func=GetCarlssonCoordinatesFeature)
# 0.5664444444444443
# classification(t=5,base_estimator='RF',n_estimators=300,func=GetCarlssonCoordinatesFeature)
# 0.9066666666666668
# classification(t=6,base_estimator='RF',n_estimators=50,func=GetCarlssonCoordinatesFeature)
# 0.8647777777777778
# classification(t=7,base_estimator='RF',n_estimators=50,func=GetCarlssonCoordinatesFeature)
# 0.8598888888888889
# classification(t=8,base_estimator='RF',n_estimators=100,func=GetCarlssonCoordinatesFeature)
# 0.7372222222222223
# classification(t=9,base_estimator='RF',n_estimators=100,func=GetCarlssonCoordinatesFeature)
# 0.7372222222222223
# classification(t=10,base_estimator='RF',n_estimators=200,func=GetCarlssonCoordinatesFeature)
# 0.6815555555555557

# print('GetPersEntropyFeature')
# classification(t=1, str_p='200', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersEntropyFeature)
# 0.47777777777777775
# classification(t=2, str_p='100', base_estimator='SVM', kernel='poly', C=223.1245475353748, degree=3, gamma=0.005940143639429267, func=GetPersEntropyFeature)
# 0.3222222222222223
# classification(t=3, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersEntropyFeature)
# 0.4696666666666667
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.5356666666666665
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.6553333333333334
# classification(t=6, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.7166666666666666
# classification(t=7, str_p='200', base_estimator='RF', n_estimators=100, func=GetPersEntropyFeature)
# 0.6606666666666667
# classification(t=8, str_p='200', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersEntropyFeature)
# 0.5777777777777777
# classification(t=9, str_p='50', base_estimator='SVM', kernel='rbf', C=879.1425034294132, gamma=0.0010352534930954075, func=GetPersEntropyFeature)
# 0.4666666666666667
# classification(t=10, str_p='200', n_estimators=500, func=GetPersEntropyFeature)
# 0.3868888888888889

# print('GetBettiCurveFeature')
# classification(t=1, str_p='200', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetBettiCurveFeature)
# 0.5444444444444444
# classification(t=2, str_p='200', base_estimator='SVM', kernel='rbf', C=288.77533858634877, gamma=0.001392949093880637, func=GetBettiCurveFeature)
# 0.5444444444444444
# classification(t=3, str_p='200', base_estimator='SVM', kernel='rbf', C=288.77533858634877, gamma=0.001392949093880637, func=GetBettiCurveFeature)
# 0.7111111111111111
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# 0.6631111111111113
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# 0.6804444444444445
# classification(t=6, str_p='200', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# 0.6365555555555555
# classification(t=7, str_p='200', base_estimator='RF', n_estimators=300, func=GetBettiCurveFeature)
# 0.6642222222222223
# classification(t=8, str_p='200', base_estimator='RF', n_estimators=50, func=GetBettiCurveFeature)
# 0.6335555555555556
# classification(t=9, str_p='100', base_estimator='RF', n_estimators=500, func=GetBettiCurveFeature)
# 0.5483333333333333
# classification(t=10, str_p='50', base_estimator='RF', n_estimators=50, func=GetBettiCurveFeature)
# 0.3400000000000001

# print('GetPersLifespanFeature')
# classification(t=1, str_p='200', base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345, degree=2, func=GetPersLifespanFeature)
# 0.47777777777777775
# classification(t=2, str_p='100', base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345, degree=2, func=GetPersLifespanFeature)
# 0.4
# classification(t=3, str_p='200', base_estimator='RF', n_estimators=300, func=GetPersLifespanFeature)
# 0.5885555555555555
# classification(t=4, str_p='200', base_estimator='RF', n_estimators=50, func=GetPersLifespanFeature)
# 0.6405555555555558
# classification(t=5, str_p='100', base_estimator='RF', n_estimators=300, func=GetPersLifespanFeature)
# 0.7620000000000001
# classification(t=6, str_p='100', base_estimator='RF', n_estimators=500, func=GetPersLifespanFeature)
# 0.8518888888888889
# classification(t=7, str_p='200', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetPersLifespanFeature)
# 0.8666666666666668
# classification(t=8, str_p='50', base_estimator='SVM', kernel='rbf', C=835.625671897373, gamma=0.00018457575175565604, func=GetPersLifespanFeature)
# 0.8222222222222222
# classification(t=9, str_p='50', base_estimator='SVM', kernel='rbf', C=141.38693859523377, func=GetPersLifespanFeature)
# 0.5888888888888888
# classification(t=10, str_p='200', base_estimator='RF', n_estimators=500, func=GetPersLifespanFeature)
# 0.545111111111111

# print('GetPersImageFeature')
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
# classification(t=1, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=100)
# 0.31244444444444447
# classification(t=2, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=200)
# 0.15599999999999997
# classification(t=3, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=50)
# 0.19088888888888889
# classification(t=4, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=300)
# 0.3562222222222222
# classification(t=5, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=500)
# 0.6131111111111109
# classification(t=6, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=100)
# 0.6915555555555557
# classification(t=7, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=200)
# 0.6705555555555557
# classification(t=8, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=100)
# 0.745777777777778
# classification(t=9, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=100)
# 0.6052222222222222
# classification(t=10, func=GetTopologicalVectorFeature, str_p='20', base_estimator='RF', n_estimators=100)
# 0.44555555555555565

# print('GetAtolFeature')
# classification(t=1, func=GetAtolFeature, str_p='8', base_estimator='RF', n_estimators=300)
# 0.7524444444444444
# classification(t=2, func=GetAtolFeature, str_p='8', base_estimator='RF', n_estimators=200)
# 0.6133333333333332
# classification(t=3, func=GetAtolFeature, str_p='8', base_estimator='RF', n_estimators=100)
# 0.7898888888888888
# classification(t=4, func=GetAtolFeature, str_p='4', base_estimator='RF', n_estimators=100)
# 0.7211111111111113
# classification(t=5, func=GetAtolFeature, str_p='4', base_estimator='RF', n_estimators=200)
# 0.9081111111111113
# classification(t=6, func=GetAtolFeature, str_p='8', base_estimator='SVM', kernel='rbf', C=835.625671897373,  gamma=0.00018457575175565604)
# 0.8666666666666667
# classification(t=7, func=GetAtolFeature, str_p='8', base_estimator='RF', n_estimators=100)
# 0.8742222222222225
# classification(t=8, func=GetAtolFeature, str_p='8', base_estimator='RF', n_estimators=100)
# 0.8777777777777779
# classification(t=9, func=GetAtolFeature, str_p='8', base_estimator='SVM', kernel='linear', C=998.1848109388686)
# 0.8444444444444446

# print('GetPerSilhouetteFeature')
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
# 0.5439999999999999

# print('GetComplexPolynomialFeature')
# classification(t=1, str_p='10', str_q='S', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.32888888888888895
# classification(t=2, str_p='10', str_q='S', base_estimator='RF', n_estimators=100, func=GetComplexPolynomialFeature)
# 0.33299999999999996
# classification(t=3, str_p='20', str_q='S', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.37333333333333335
# classification(t=4, str_p='20', str_q='S', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.4496666666666666
# classification(t=5, str_p='20', str_q='S', base_estimator='RF', n_estimators=300, func=GetComplexPolynomialFeature)
# 0.7633333333333333
# classification(t=6, str_p='20', str_q='S', base_estimator='RF', n_estimators=300, func=GetComplexPolynomialFeature)
# 0.8263333333333334
# classification(t=7, str_p='10', str_q='S', base_estimator='RF', n_estimators=50, func=GetComplexPolynomialFeature)
# 0.8234444444444445
# classification(t=8, str_p='20', str_q='T', base_estimator='RF', n_estimators=100, func=GetComplexPolynomialFeature)
# 0.8519999999999999
# classification(t=9, str_p='10', str_q='T', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.771111111111111
# classification(t=10, str_p='20', str_q='T', base_estimator='RF', n_estimators=200, func=GetComplexPolynomialFeature)
# 0.6972222222222223

# print('GetPersLandscapeFeature')
# classification(t=1, str_p='200', str_q='5', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersLandscapeFeature)
# 0.5666666666666667
# classification(t=2, str_p='50', str_q='20', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersLandscapeFeature)
# 0.6222222222222221
# classification(t=3, str_p='200', str_q='20', base_estimator='SVM', kernel='poly', C=1000.0405153241447, gamma=0.0009688387165373345,  degree=2, func=GetPersLandscapeFeature)
# 0.7222222222222222
# classification(t=4, str_p='100', str_q='10', base_estimator='SVM', kernel='linear', C=998.1848109388686, func=GetPersLandscapeFeature)
# 0.7222222222222222
# classification(t=5, str_p='200', str_q='20', base_estimator='RF', n_estimators=50, func=GetPersLandscapeFeature)
# 0.8888888888888892
# classification(t=6, str_p='50', str_q='20', base_estimator='RF', n_estimators=50, func=GetPersLandscapeFeature)
# 0.8718888888888892
# classification(t=7, str_p='200', str_q='2', base_estimator='RF', n_estimators=100, func=GetPersLandscapeFeature)
# 0.8436666666666666
# classification(t=8, str_p='200', str_q='5', base_estimator='SVM', kernel='rbf', C=212.62811600005904, gamma=0.0030862881073963535, func=GetPersLandscapeFeature)
# 0.8555555555555555
# classification(t=9, str_p='200', str_q='10', base_estimator='RF', n_estimators=300, func=GetPersLandscapeFeature)
# 0.8156666666666668
# classification(t=10, str_p='100', str_q='10', base_estimator='SVM', kernel='linear', C=274.0499742167474, func=GetPersLandscapeFeature)
# 0.5

# print('GetAdaptativeSystemFeature')
#classification(t=1, base_estimator='RF', n_estimators=100, str_p='gmm', str_q='30', func=GetAdaptativeSystemFeature)
# 0.7951111111111111
# classification(t=2, base_estimator='RF', n_estimators=100, str_p='gmm', str_q='25', func=GetAdaptativeSystemFeature)
# 0.8160000000000003
# classification(t=3, base_estimator='RF', n_estimators=300, str_p='gmm', str_q='25', func=GetAdaptativeSystemFeature)
# 0.7836666666666667
# classification(t=4, base_estimator='RF', n_estimators=300, str_p='gmm', str_q='15', func=GetAdaptativeSystemFeature)
# 0.7275555555555556
# classification(t=5, base_estimator='RF', n_estimators=500, str_p='gmm', str_q='10', func=GetAdaptativeSystemFeature)
# 0.932
# classification(t=6, base_estimator='SVM', kernel='rbf', C=147.72857490581015, gamma=0.008899057397352507, str_p='gmm', str_q='15', func=GetAdaptativeSystemFeature)
# 0.8666666666666667
# classification(t=7, base_estimator='RF', n_estimators=300, str_p='gmm', str_q='10', func=GetAdaptativeSystemFeature)
# 0.8814444444444447
# classification(t=8, base_estimator='SVM', kernel='rbf', C=294.6141483736795, gamma=0.0033936188164774834, str_p='gmm', str_q='4', func=GetAdaptativeSystemFeature)
# 0.8777777777777781
# classification(t=9, base_estimator='RF', n_estimators=50, str_p='gmm', str_q='10', func=GetAdaptativeSystemFeature)
# 0.7044444444444444
# classification(t=10, base_estimator='RF', n_estimators=300, str_p='gmm', str_q='5', func=GetAdaptativeSystemFeature)
# 0.5723333333333331

# print('GetTemplateFunctionFeature')
# classification(t=1, base_estimator='SVM', str_p='15', str_q='1.2', C=998.1848109388686, kernel='linear', func=GetTemplateFunctionFeature)
# 0.511
# classification(t=2, base_estimator='SVM', str_p='12', str_q='0.6', C=998.1848109388686, kernel='linear', func=GetTemplateFunctionFeature)
# 0.6
# classification(t=3, base_estimator='SVM', str_p='14', str_q='0.9', kernel='rbf', C=879.1425034294132, gamma=0.0010352534930954075, func=GetTemplateFunctionFeature)
# 0.722
# classification(t=4, base_estimator='SVM', str_p='6', str_q='0.5', C=141.38693859523377, kernel='linear', func=GetTemplateFunctionFeature)
# 0.722
# classification(t=5, str_p='14', str_q='1', base_estimator='RF', n_estimators=100, func=GetTemplateFunctionFeature)
# 0.898
# classification(t=6, base_estimator='RF', str_p='12', str_q='0.7', n_estimators=500, func=GetTemplateFunctionFeature)
# 0.919
# classification(t=7, base_estimator='SVM', str_p='12', str_q='0.9', C=288.77533858634877, kernel='rbf', gamma=0.001392949093880637, func=GetTemplateFunctionFeature)
# 0.911
# classification(t=8, base_estimator='SVM', str_p='14', str_q='1.2', kernel='rbf', C=288.77533858634877, gamma=0.001392949093880637, func=GetTemplateFunctionFeature)
# 0.944
# classification(t=9, base_estimator='RF', str_p='6', str_q='1.2', n_estimators=500, func=GetTemplateFunctionFeature)
# 0.744

# print('GetAdaptativeSystemFeature')
# classification(t=1, base_estimator='SVM', str_p='gmm', str_q='20',
#                C=97.17226044546167, kernel='linear', func=GetAdaptativeSystemFeature)

# print('GetPersTropicalCoordinatesFeature')
# classification(t=1, base_estimator='RF', n_estimators=200, str_p='52.19579592311136', func=GetPersTropicalCoordinatesFeature)
# 0.4556666666666668
# classification(t=2, base_estimator='RF', n_estimators=100, str_p='19.234452089092336', func=GetPersTropicalCoordinatesFeature)
# 0.4222222222222222
# classification(t=3, base_estimator='RF', n_estimators=100, str_p='19.234452089092336', func=GetPersTropicalCoordinatesFeature)
# 0.4918888888888888
# classification(t=4, base_estimator='RF', n_estimators=200, str_p='52.19579592311136', func=GetPersTropicalCoordinatesFeature)
# 0.6073333333333334
# classification(t=5, base_estimator='RF', n_estimators=200, str_p='166.92513437947457', func=GetPersTropicalCoordinatesFeature)
# 0.8692222222222225
# classification(t=6, base_estimator='RF', n_estimators=500, str_p='104.95987537408911', func=GetPersTropicalCoordinatesFeature)
# 0.8622222222222222
# classification(t=7, base_estimator='RF', n_estimators=50, str_p='1.801599392208586', func=GetPersTropicalCoordinatesFeature)
# 0.8384444444444443
# classification(t=8, base_estimator='RF', n_estimators=500, str_p='165.82938147342833', func=GetPersTropicalCoordinatesFeature)
# 0.7772222222222223
# classification(t=9, base_estimator='RF', n_estimators=50, str_p='188.3673897506845', func=GetPersTropicalCoordinatesFeature)
# 0.7453333333333333
# classification(t=10, base_estimator='SVM', C=429.0911898712949, kernel='rbf', gamma=0.033478475471326576, str_p='132.6882995636896', func=GetPersTropicalCoordinatesFeature)
# 0.6



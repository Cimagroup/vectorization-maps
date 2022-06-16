import pickle
import numpy as np
from fashion_mnist import mnist_reader
from sklearn.ensemble import RandomForestClassifier
from vectorisation import *

path_feat = "fashion_mnist/features/"
path_diag = "fashion_mnist/pdiagrams/"
path_res = "fashion_mnist/results/"

n_total = 70000

func_list = [
            GetPersStats,
            GetPersImageFeature,
            GetPersLandscapeFeature,
            GetPersEntropyFeature,
            GetBettiCurveFeature,
            GetCarlssonCoordinatesFeature,
            GetPersSilhouetteFeature,
            GetTopologicalVectorFeature,
            #GetAtolFeature,
            GetComplexPolynomialFeature,
            GetPersLifespanFeature,
            GetPersTropicalCoordinatesFeature
            ]

#%%

with open(path_feat + 'taxi_d0.pkl', 'rb') as f:
    features_taxi_d0 = pickle.load(f)
with open(path_feat + 'taxi_d1.pkl', 'rb') as f:
    features_taxi_d1 = pickle.load(f)
with open(path_feat + 'euc_d0.pkl', 'rb') as f:
    features_euc_d0 = pickle.load(f)
with open(path_feat + 'euc_d1.pkl', 'rb') as f:
    features_euc_d1 = pickle.load(f)

features_taxi = {}
features_euc = {}

for func in func_list:
    for i in range(n_total):
        features_taxi[(func.__name__)+'_'+str(i)]=np.hstack([features_taxi_d0[(func.__name__)+'_'+str(i)],
                                                             features_taxi_d1[(func.__name__)+'_'+str(i)]])
        features_euc[(func.__name__)+'_'+str(i)]=np.hstack([features_euc_d0[(func.__name__)+'_'+str(i)],
                                                             features_euc_d1[(func.__name__)+'_'+str(i)]])
#%%

_, train_labels = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
_, test_labels = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

train_index = range(60000)
test_index = range(60000,70000)

def classification(train_labels, test_labels, train_index, test_index, func_list, features):
    train_scores = dict()
    test_scores = dict()
    for func in func_list:
        print(func.__name__)
        X = []
        Y = []
        for i in train_index:
            name = (func.__name__)+'_'+str(i)
            X.append(features[name])
        clf = RandomForestClassifier()
        clf = clf.fit(X, train_labels)      
        train_scores[func] = clf.score(X, train_labels)
        for i in test_index:
            name = (func.__name__)+'_'+str(i)
            Y.append(features[name])
        test_scores[func] = clf.score(Y, test_labels)
        
    return train_scores, test_scores
 
#%%
print('----| EUCLIDEAN |----')
n = 10
scores_vector_euc = np.zeros([n, len(func_list)])
for i in range(n):
    print('----| ', str(i), ' |----')
    _, test_scores_euc = classification(train_labels=train_labels, 
                                         test_labels=test_labels, 
                                         train_index=train_index, 
                                         test_index=test_index, 
                                         func_list=func_list, 
                                         features=features_euc)
    scores_vector_euc[i, :] = np.array([x[1] for x in list(test_scores_euc.items())])
    
scores_avg_euc = dict()
for j in range(len(func_list)):
    scores_avg_euc[func_list[j].__name__] = (scores_vector_euc[:,j].mean(), scores_vector_euc[:,j].std())

print(scores_avg_euc)

with open(path_res + 'fashionMNIST_euc_edge_scores.pkl', 'wb') as f:
  pickle.dump(scores_avg_euc, f)

#%%
print('----| TAXI |----')
n = 10
scores_vector_taxi = np.zeros([n, len(func_list)])
for i in range(n):
    print('----| ', str(i), ' |----')
    _, test_scores_taxi = classification(train_labels=train_labels, 
                                         test_labels=test_labels, 
                                         train_index=train_index, 
                                         test_index=test_index, 
                                         func_list=func_list, 
                                         features=features_taxi)
    scores_vector_taxi[i, :] = np.array([x[1] for x in list(test_scores_taxi.items())])
    
scores_avg_taxi = dict()
for j in range(len(func_list)):
    scores_avg_taxi[func_list[j].__name__] = (scores_vector_taxi[:,j].mean(), scores_vector_taxi[:,j].std())

print(scores_avg_taxi)

with open(path_res + 'fashionMNIST_taxi_edge_scores.pkl', 'wb') as f:
  pickle.dump(scores_avg_taxi, f)

#{'GetPersStats': (0.15287999999999996, 0.0010486181383134638), 'GetPersImageFeature': (0.15239, 0.0009741149829460606), 'GetPersLandscapeFeature': (0.11749999999999998, 1.3877787807814457e-17), 'GetPersEntropyFeature': (0.10784, 0.00021540659228538016), 'GetBettiCurveFeature': (0.15282, 0.001028396810574595), 'GetCarlssonCoordinatesFeature': (0.10744000000000001, 0.0002497999199359352), 'GetPersSilhouetteFeature': (0.10735000000000001, 0.00022022715545545274), 'GetTopologicalVectorFeature': (0.10838, 0.0003310589071449376), 'GetComplexPolynomialFeature': (0.15281999999999998, 0.0008388086790204251), 'GetPersLifespanFeature': (0.10728, 0.00024413111231467562), 'GetPersTropicalCoordinatesFeature': (0.15289999999999998, 0.0009497368056467031)}
#{'GetPersStats': (0.15284999999999999, 0.001032714868683506), 'GetPersImageFeature': (0.15289, 0.0008791473141629873), 'GetPersLandscapeFeature': (0.11749999999999998, 1.3877787807814457e-17), 'GetPersEntropyFeature': (0.10775000000000001, 0.00023345235059857666), 'GetBettiCurveFeature': (0.15285, 0.0010442700800080361), 'GetCarlssonCoordinatesFeature': (0.10761000000000001, 0.00024269322199023253), 'GetPersSilhouetteFeature': (0.10727, 0.00028999999999999783), 'GetTopologicalVectorFeature': (0.10824, 0.0002835489375751585), 'GetComplexPolynomialFeature': (0.15297, 0.0009295697929687711), 'GetPersLifespanFeature': (0.1076, 0.00026832815729997315), 'GetPersTropicalCoordinatesFeature': (0.15280999999999997, 0.0011004090148667401)}
  

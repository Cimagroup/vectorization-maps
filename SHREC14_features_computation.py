import pickle
import numpy as np
from vectorisation import *
import pandas as pd

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

#%%
#Methods with no parameter

func_list = [
            # GetPersStats,
            # GetCarlssonCoordinatesFeature
            ]

for func in func_list:
    features={}
    print(func.__name__)
    for t in range(1,12):
        dgms = dgmsDF[dgmsDF.trial==t]
        dgms = np.array(dgms['Dgm1'])
        
        for i in range(dgms.shape[0]):
            features[str(t)+'_'+str(i)]=func(dgms[i]) 
        
    with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
      pickle.dump(features, f)
      
#%%
#Methods with only one parameter
func_list = [
             # GetPersEntropyFeature,
             # GetBettiCurveFeature,
              GetTopologicalVectorFeature,
             # GetPersLifespanFeature,
              GetPersImageFeature,
             # GetAtolFeature
            ]

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetPersImageFeature'] = [10,25,50,100,200]
hyper_parameters['GetAtolFeature'] = [2,4,8]


for func in func_list:
    features={}
    print(func.__name__)
    for p in hyper_parameters[func.__name__]:
        print(str(p))
        for t in range(1,12):
            print(t)
            dgms = dgmsDF[dgmsDF.trial==t]
            dgms = np.array(dgms['Dgm1'])
            for i in range(dgms.shape[0]):
                features[str(t)+'_'+str(i)+'_'+str(p)]=func(dgms[i],p) 
            
        with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
          pickle.dump(features, f)
          
#%%
#Methods with two parameter
func_list = [
               # GetPersSilhouetteFeature,
              GetComplexPolynomialFeature,
              GetPersLandscapeFeature
            ]

hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [1,2,3,5,10,20]]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]

for func in func_list:
    features={}
    print(func.__name__)
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print(str(p),str(q))
            for t in range(1,12):
                print(t)
                dgms = dgmsDF[dgmsDF.trial==t]
                dgms = np.array(dgms['Dgm1'])
                for i in range(dgms.shape[0]):
                    features[str(t)+'_'+str(i)+'_'+str(p)+'_'+str(q)]=func(dgms[i],p,q) 
                
            with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
              pickle.dump(features, f)



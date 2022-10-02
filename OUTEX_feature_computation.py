import pickle
import numpy as np
from vectorisation import *


#%%
#Load_barcodes
pdiagrams = dict()
path_diag = "Outex-TC-00024/pdiagrams/"
path_feat = "Outex-TC-00024/features/"

#Barcodes with just one bar are loaded as a 1d-array.
#We force them to be a 2d-array
def safe_load(x):
    pd = np.loadtxt(x)
    if (len(pd.shape)==1) and (pd.shape[0]>0): 
        pd = pd.reshape(1,2)
    return pd

for i in range(2720):
    pdiagrams["pdiag_l_d0_"+str(i)]= safe_load(path_diag + "l_d0_"+str(i))
    pdiagrams["pdiag_l_d1_"+str(i)]= safe_load(path_diag + "l_d1_"+str(i))
    pdiagrams["pdiag_u_d0_"+str(i)]= safe_load(path_diag + "u_d0_"+str(i))
    pdiagrams["pdiag_u_d1_"+str(i)]= safe_load(path_diag + "u_d1_"+str(i))
    
#%%

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetAtolFeature'] = [2,4,8,16,32,64]
hyper_parameters['GetPersImageFeature'] = [25,50,100,150,200]
hyper_parameters['GetPersSilhouetteFeature'] = [[50,100,200], [0,1,2,5,10,20]]
hyper_parameters['GetComplexPolynomialFeature'] = [[5, 10, 20],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[50,100,200], [2,5,10,20]]

#%%
#Methods with no parameter
func_list = [
             # GetPersStats,
             # GetCarlssonCoordinatesFeature
            ]

for func in func_list:
    features_l_d0 = dict()
    features_l_d1 = dict()
    features_u_d0 = dict()
    features_u_d1 = dict()

    print(func.__name__)
    for i in range(2720):
        features_l_d0[str(i)]=func(pdiagrams["pdiag_l_d0_"+str(i)])
        features_l_d1[str(i)]=func(pdiagrams["pdiag_l_d1_"+str(i)])
        features_u_d0[str(i)]=func(pdiagrams["pdiag_u_d0_"+str(i)])
        features_u_d1[str(i)]=func(pdiagrams["pdiag_u_d1_"+str(i)])    
        
    with open(path_feat + func.__name__ +'_l_d0.pkl', 'wb') as f:
      pickle.dump(features_l_d0, f)
    with open(path_feat + func.__name__ +'_l_d1.pkl', 'wb') as f:
      pickle.dump(features_l_d1, f)
    with open(path_feat + func.__name__ +'_u_d0.pkl', 'wb') as f:
      pickle.dump(features_u_d0, f)
    with open(path_feat + func.__name__ +'_u_d1.pkl', 'wb') as f:
      pickle.dump(features_u_d1, f)

#%%
#Methods with only one parameter
func_list = [
              # GetPersEntropyFeature,
              # GetBettiCurveFeature,
              # GetTopologicalVectorFeature,
              # GetPersLifespanFeature,
              GetPersImageFeature
            ]

for func in func_list:
    features_l_d0 = dict()
    features_l_d1 = dict()
    features_u_d0 = dict()
    features_u_d1 = dict()

    print(func.__name__)
    for p in hyper_parameters[func.__name__]:
        for i in range(2720):
            features_l_d0[str(i)+'_'+str(p)]=func(pdiagrams["pdiag_l_d0_"+str(i)],p)
            features_l_d1[str(i)+'_'+str(p)]=func(pdiagrams["pdiag_l_d1_"+str(i)],p)
            features_u_d0[str(i)+'_'+str(p)]=func(pdiagrams["pdiag_u_d0_"+str(i)],p)
            features_u_d1[str(i)+'_'+str(p)]=func(pdiagrams["pdiag_u_d1_"+str(i)],p)    
        
    with open(path_feat + func.__name__ +'_l_d0.pkl', 'wb') as f:
      pickle.dump(features_l_d0, f)
    with open(path_feat + func.__name__ +'_l_d1.pkl', 'wb') as f:
      pickle.dump(features_l_d1, f)
    with open(path_feat + func.__name__ +'_u_d0.pkl', 'wb') as f:
      pickle.dump(features_u_d0, f)
    with open(path_feat + func.__name__ +'_u_d1.pkl', 'wb') as f:
      pickle.dump(features_u_d1, f)
      
#%%
func = GetAtolFeature


list_l_d0 = []
list_l_d1 = []
list_u_d0 = []
list_u_d1 = []
for i in range(2720):
    list_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    list_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    list_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    list_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])
    
features_l_d0 = dict()
features_l_d1 = dict()
features_u_d0 = dict()
features_u_d1 = dict()
for p in hyper_parameters[func.__name__]:    
    print(p)
    
    atol_l_d0 = func(list_l_d0, p)
    atol_l_d1 = func(list_l_d1, p)
    atol_u_d0 = func(list_u_d0, p)
    atol_u_d1 = func(list_u_d1, p)
    
    for i in range(2720):
        features_l_d0[str(i)+'_'+str(p)]=atol_l_d0[i,:]
        features_l_d1[str(i)+'_'+str(p)]=atol_l_d1[i,:]
        features_u_d0[str(i)+'_'+str(p)]=atol_u_d0[i,:]
        features_u_d1[str(i)+'_'+str(p)]=atol_u_d1[i,:]

with open(path_feat + func.__name__ +'_l_d0.pkl', 'wb') as f:
  pickle.dump(features_l_d0, f)
with open(path_feat + func.__name__ +'_l_d1.pkl', 'wb') as f:
  pickle.dump(features_l_d1, f)
with open(path_feat + func.__name__ +'_u_d0.pkl', 'wb') as f:
  pickle.dump(features_u_d0, f)
with open(path_feat + func.__name__ +'_u_d1.pkl', 'wb') as f:
  pickle.dump(features_u_d1, f)    
        

#%%
#Methods with two parameter
func_list = [
             # GetPersSilhouetteFeature,
             # GetComplexPolynomialFeature,
             # GetPersLandscapeFeature
            ]

for func in func_list:
    features_l_d0 = dict()
    features_l_d1 = dict()
    features_u_d0 = dict()
    features_u_d1 = dict()
    
    print(func.__name__)
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print(p,q)
            for i in range(2720):
                features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=func(pdiagrams["pdiag_l_d0_"+str(i)],p,q)
                features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=func(pdiagrams["pdiag_l_d1_"+str(i)],p,q)
                features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=func(pdiagrams["pdiag_u_d0_"+str(i)],p,q)
                features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=func(pdiagrams["pdiag_u_d1_"+str(i)],p,q)

                    
    with open(path_feat + func.__name__ +'_l_d0.pkl', 'wb') as f:
      pickle.dump(features_l_d0, f)
    with open(path_feat + func.__name__ +'_l_d1.pkl', 'wb') as f:
      pickle.dump(features_l_d1, f)
    with open(path_feat + func.__name__ +'_u_d0.pkl', 'wb') as f:
      pickle.dump(features_u_d0, f)
    with open(path_feat + func.__name__ +'_u_d1.pkl', 'wb') as f:
      pickle.dump(features_u_d1, f)

#%%
print('DONE')


import pickle
import numpy as np
from fashion_mnist import mnist_reader
from vectorisation import *

path_feat = "fashion_mnist/features/"
path_diag= "fashion_mnist/pdiagrams/"


#%%
_, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
_, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

n_train = len(y_train)
n_total = len(y_train) + len(y_test)

#%%
pdiagrams = dict()

#Barcodes with just one bar are loaded as a 1d-array.
#We force them to be a 2d-array
def safe_load(x):
    pd = np.loadtxt(x)
    if (len(pd.shape)==1) and (pd.shape[0]>0): 
        pd = pd.reshape(1,2)
    return pd

for i in range(n_total):
    pdiagrams["pdiag_taxi_l_"+str(i)]= safe_load(path_diag + "taxi_l_"+str(i))
    pdiagrams["pdiag_taxi_u_"+str(i)]= safe_load(path_diag + "taxi_u_"+str(i))
    
#%%

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [15,30,50]
hyper_parameters['GetBettiCurveFeature'] = [15,30,50]
hyper_parameters['GetPersLifespanFeature'] = [15,30,50]
hyper_parameters['GetTopologicalVectorFeature'] = [3, 5, 10]
hyper_parameters['GetAtolFeature'] = [2,4,8,16]
hyper_parameters['GetPersImageFeature'] = [3,6,12,20]
hyper_parameters['GetPersSilhouetteFeature'] = [[15,30,50], [0,1,2,5]]
hyper_parameters['GetComplexPolynomialFeature'] = [[3, 5, 10],['R', 'S', 'T']]
hyper_parameters['GetPersLandscapeFeature'] = [[15,30,50], [1,2,3,5]]
hyper_parameters['GetTemplateFunctionFeature'] = [[1,2,3,5], [.5, 1, 2]]
hyper_parameters['GetAdaptativeSystemFeature'] = [['gmm', 'hdb'], 
                                                [1,2,3,4,5,10,15]]

#%%
#Methods with no parameter
func_list = [
             GetPersStats,
             GetCarlssonCoordinatesFeature
            ]

for func in func_list:
    features_l = dict()
    features_u = dict()

    print(func.__name__)
    for i in range(n_total):
        features_l[str(i)]=func(pdiagrams["pdiag_taxi_l_"+str(i)])
        features_u[str(i)]=func(pdiagrams["pdiag_taxi_u_"+str(i)])
        
    with open(path_feat + func.__name__ +'_l.pkl', 'wb') as f:
      pickle.dump(features_l, f)
    with open(path_feat + func.__name__ +'_u.pkl', 'wb') as f:
      pickle.dump(features_u, f)
        
#%%
#Methods with only one parameter
func_list = [
              GetPersEntropyFeature,
              GetBettiCurveFeature,
              GetTopologicalVectorFeature,
              GetPersLifespanFeature,
              GetPersImageFeature
            ]

for func in func_list:
    features_l = dict()
    features_u = dict()

    print(func.__name__)
    for p in hyper_parameters[func.__name__]:
        for i in range(n_total):
            features_l[str(i)+'_'+str(p)]=func(pdiagrams["pdiag_taxi_l_"+str(i)],p)
            features_u[str(i)+'_'+str(p)]=func(pdiagrams["pdiag_taxi_u_"+str(i)],p)
            
    with open(path_feat + func.__name__ +'_l.pkl', 'wb') as f:
      pickle.dump(features_l, f)
    with open(path_feat + func.__name__ +'_u.pkl', 'wb') as f:
      pickle.dump(features_u, f)
      
#%%
func = GetAtolFeature
    
features_l = dict()
features_u = dict()
for p in hyper_parameters[func.__name__]:    
    print(p)
    
    atol_l = func(list_l, p)
    atol_u = func(list_u, p)
    
    for i in range(n_total):
        features_l[str(i)+'_'+str(p)]=atol_l[i,:]
        features_u[str(i)+'_'+str(p)]=atol_u[i,:]

with open(path_feat + func.__name__ +'_l.pkl', 'wb') as f:
  pickle.dump(features_l, f)
with open(path_feat + func.__name__ +'_u.pkl', 'wb') as f:
  pickle.dump(features_u, f)    

#%%
#Methods with two parameter
func_list = [
              GetPersSilhouetteFeature,
              GetComplexPolynomialFeature,
              GetPersLandscapeFeature
            ]

for func in func_list:
    features_l = dict()
    features_u = dict()
    
    print(func.__name__)
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print(p,q)
            for i in range(n_total):
                features_l[str(i)+'_'+str(p)+'_'+str(q)]=func(pdiagrams["pdiag_taxi_l_"+str(i)],p,q)
                features_u[str(i)+'_'+str(p)+'_'+str(q)]=func(pdiagrams["pdiag_taxi_u_"+str(i)],p,q)

                    
    with open(path_feat + func.__name__ +'_l.pkl', 'wb') as f:
      pickle.dump(features_l, f)
    with open(path_feat + func.__name__ +'_u.pkl', 'wb') as f:
      pickle.dump(features_u, f)
          
#%%

func = GetTemplateFunctionFeature
    
print(func.__name__)

train_l = []
train_u = []

for i in range(n_train):
    train_l.append(pdiagrams["pdiag_taxi_l_"+str(i)])
    train_u.append(pdiagrams["pdiag_taxi_u_"+str(i)])
    
test_l = []
test_u = []

for i in range(n_train, n_total):
    test_l.append(pdiagrams["pdiag_taxi_l_"+str(i)])
    test_u.append(pdiagrams["pdiag_taxi_u_"+str(i)])

    
features_l = dict()
features_u = dict()

for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        print(p,q)
        
        tent_l = func(train_l, test_l, 
                         d=p, padding=q)
        tent_u = func(train_u, test_u, 
                         d=p, padding=q)

        for i in range(n_total):
            j = Z.index(i)
            features_l[str(i)+'_'+str(p)+'_'+str(q)]=tent_l[j,:]
            features_u[str(i)+'_'+str(p)+'_'+str(q)]=tent_u[j,:]
            
with open(path_feat + func.__name__ +'_l.pkl', 'wb') as f:
  pickle.dump(features_l, f)
with open(path_feat + func.__name__ +'_u.pkl', 'wb') as f:
  pickle.dump(features_u, f)

#%%

func = GetAdaptativeSystemFeature
    
print(func.__name__)

train_l = []
train_u = []

for i in range(n_train):
    train_l.append(pdiagrams["pdiag_taxi_l_"+str(i)])
    train_u.append(pdiagrams["pdiag_taxi_u_"+str(i)])
    
test_l = []
test_u = []

for i in range(n_train, n_total):
    test_l.append(pdiagrams["pdiag_taxi_l_"+str(i)])
    test_u.append(pdiagrams["pdiag_taxi_u_"+str(i)])

    
features_l = dict()
features_u = dict()

for p in hyper_parameters[func.__name__][0]:
    if p=='gmm':
        for q in hyper_parameters[func.__name__][1]:
            print(p,q)
            
            ats_l = func(train_l, test_l, 
                         y_train = y_train, model=p, d=q)
            ats_u = func(train_u, test_u, 
                         y_train = y_train, model=p, d=q)
    
            for i in range(n_total):
                j = Z.index(i)
                features_l[str(i)+'_'+str(p)+'_'+str(q)]=ats_l[j,:]
                features_u[str(i)+'_'+str(p)+'_'+str(q)]=ats_u[j,:]
    else:
        q=1
        ats_l = func(train_l, test_l, 
                     y_train = y_train, model=p, d=q)
        ats_u = func(train_u, test_u, 
                     y_train = y_train, model=p, d=q)

        for i in range(n_total):
            j = Z.index(i)
            features_l[str(i)+'_'+str(p)+'_'+str(q)]=ats_l[j,:]
            features_u[str(i)+'_'+str(p)+'_'+str(q)]=ats_u[j,:]
        
            
with open(path_feat + func.__name__ +'_l.pkl', 'wb') as f:
  pickle.dump(features_l, f)
with open(path_feat + func.__name__ +'_u.pkl', 'wb') as f:
  pickle.dump(features_u, f)


#%%
print('DONE')


import pickle
import numpy as np
import pandas as pd
from vectorisation import *
from sklearn.model_selection import train_test_split


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
hyper_parameters['GetTentFunctionFeature'] = [[50,100,200], [1,3,5]]
hyper_parameters['GetTemplateSystemFeature'] = [['gmm', 'hdb'], 
                                                [1,2,3,4, 5,10,15,20,25,30,35,40,45,50]]

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
              # GetPersImageFeature
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
# func = GetAtolFeature


# list_l_d0 = []
# list_l_d1 = []
# list_u_d0 = []
# list_u_d1 = []
# for i in range(2720):
#     list_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
#     list_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
#     list_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
#     list_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])
    
# features_l_d0 = dict()
# features_l_d1 = dict()
# features_u_d0 = dict()
# features_u_d1 = dict()
# for p in hyper_parameters[func.__name__]:    
#     print(p)
    
#     atol_l_d0 = func(list_l_d0, p)
#     atol_l_d1 = func(list_l_d1, p)
#     atol_u_d0 = func(list_u_d0, p)
#     atol_u_d1 = func(list_u_d1, p)
    
#     for i in range(2720):
#         features_l_d0[str(i)+'_'+str(p)]=atol_l_d0[i,:]
#         features_l_d1[str(i)+'_'+str(p)]=atol_l_d1[i,:]
#         features_u_d0[str(i)+'_'+str(p)]=atol_u_d0[i,:]
#         features_u_d1[str(i)+'_'+str(p)]=atol_u_d1[i,:]

# with open(path_feat + func.__name__ +'_l_d0.pkl', 'wb') as f:
#   pickle.dump(features_l_d0, f)
# with open(path_feat + func.__name__ +'_l_d1.pkl', 'wb') as f:
#   pickle.dump(features_l_d1, f)
# with open(path_feat + func.__name__ +'_u_d0.pkl', 'wb') as f:
#   pickle.dump(features_u_d0, f)
# with open(path_feat + func.__name__ +'_u_d1.pkl', 'wb') as f:
#   pickle.dump(features_u_d1, f)    
        

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
# To calculate the tent functions and adaptative template systems, we need
# to separate the training set from the test set

path_feat = 'Outex-TC-00024/features/'
path_data = "Outex-TC-00024/data/000/"

train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                    usecols=[1]).to_numpy().flatten().tolist())
test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                   usecols=[1]).to_numpy().flatten().tolist())

#This is the OUTEX-68 case
labels = np.hstack([train_labels, test_labels])
Z_train_68, Z_test_68, y_train_68, y_test_68 = train_test_split(range(2720), labels, 
                                                                test_size=0.3, 
                                                                random_state=0)

# And the OUTEX-10 case
from numpy.random import seed, choice
seed(1)
labels = range(68)
labels = choice(labels, size=(10), replace = False)

train_indexes = np.array([i for i in range(len(train_labels)) if train_labels[i] in labels])
test_indexes = np.array([i for i in range(len(test_labels)) if test_labels[i] in labels])
label_list = np.hstack([train_labels[train_indexes], test_labels[test_indexes]])


Z_train_10, Z_test_10, y_train_10, y_test_10 = train_test_split(range(len(label_list)), 
                                                                label_list, test_size=0.3, 
                                                                random_state=0)


#%%

func = GetTentFunctionFeature
    
print(func.__name__)
print(68)

train_l_d0 = []
train_l_d1 = []
train_u_d0 = []
train_u_d1 = []

for i in Z_train_68:
    train_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    train_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    train_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    train_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])
    
test_l_d0 = []
test_l_d1 = []
test_u_d0 = []
test_u_d1 = []

for i in Z_test_68:
    test_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    test_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    test_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    test_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])

    
features_l_d0 = dict()
features_l_d1 = dict()
features_u_d0 = dict()
features_u_d1 = dict()

for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        print(p,q)
        
        tent_l_d0 = func(train_l_d0, test_l_d0, 
                         d=p, padding=q)
        tent_l_d1 = func(train_l_d1, test_l_d1, 
                         d=p, padding=q)
        tent_u_d0 = func(train_u_d0, test_u_d0, 
                         d=p, padding=q)
        tent_u_d1 = func(train_u_d1, test_u_d1, 
                         d=p, padding=q)

        Z = Z_train_68 + Z_test_68
        for i in Z:
            j = Z.index(i)
            features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=tent_l_d0[j,:]
            features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=tent_u_d1[j,:]
            features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=tent_u_d0[j,:]
            features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=tent_u_d1[j,:]
            
with open(path_feat + func.__name__ +'68_l_d0.pkl', 'wb') as f:
  pickle.dump(features_l_d0, f)
with open(path_feat + func.__name__ +'68_l_d1.pkl', 'wb') as f:
  pickle.dump(features_l_d1, f)
with open(path_feat + func.__name__ +'68_u_d0.pkl', 'wb') as f:
  pickle.dump(features_u_d0, f)
with open(path_feat + func.__name__ +'68_u_d1.pkl', 'wb') as f:
  pickle.dump(features_u_d1, f)
  
#%%

func = GetTentFunctionFeature
    
print(func.__name__)
print(10)

train_l_d0 = []
train_l_d1 = []
train_u_d0 = []
train_u_d1 = []

for i in Z_train_10:
    train_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    train_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    train_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    train_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])
    
test_l_d0 = []
test_l_d1 = []
test_u_d0 = []
test_u_d1 = []

for i in Z_test_10:
    test_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    test_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    test_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    test_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])

    
features_l_d0 = dict()
features_l_d1 = dict()
features_u_d0 = dict()
features_u_d1 = dict()

for p in hyper_parameters[func.__name__][0]:
    for q in hyper_parameters[func.__name__][1]:
        print(p,q)
        
        tent_l_d0 = func(train_l_d0, test_l_d0, 
                         d=p, padding=q)
        tent_l_d1 = func(train_l_d1, test_l_d1, 
                         d=p, padding=q)
        tent_u_d0 = func(train_u_d0, test_u_d0, 
                         d=p, padding=q)
        tent_u_d1 = func(train_u_d1, test_u_d1, 
                         d=p, padding=q)

        Z = Z_train_10 + Z_test_10
        for i in Z:
            j = Z.index(i)
            features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=tent_l_d0[j,:]
            features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=tent_u_d1[j,:]
            features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=tent_u_d0[j,:]
            features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=tent_u_d1[j,:]
            
with open(path_feat + func.__name__ +'10_l_d0.pkl', 'wb') as f:
  pickle.dump(features_l_d0, f)
with open(path_feat + func.__name__ +'10_l_d1.pkl', 'wb') as f:
  pickle.dump(features_l_d1, f)
with open(path_feat + func.__name__ +'10_u_d0.pkl', 'wb') as f:
  pickle.dump(features_u_d0, f)
with open(path_feat + func.__name__ +'10_u_d1.pkl', 'wb') as f:
  pickle.dump(features_u_d1, f)
  
#%%

func = GetTemplateSystemFeature
    
print(func.__name__)
print(68)

train_l_d0 = []
train_l_d1 = []
train_u_d0 = []
train_u_d1 = []

for i in Z_train_68:
    train_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    train_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    train_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    train_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])
    
test_l_d0 = []
test_l_d1 = []
test_u_d0 = []
test_u_d1 = []

for i in Z_test_68:
    test_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    test_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    test_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    test_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])

    
features_l_d0 = dict()
features_l_d1 = dict()
features_u_d0 = dict()
features_u_d1 = dict()

for p in hyper_parameters[func.__name__][0]:
    if p=='gmm':
        for q in hyper_parameters[func.__name__][1]:
            print(p,q)
            
            ats_l_d0 = func(train_l_d0, test_l_d0, 
                             y_train=y_train_68, model=p, d=q)
            ats_l_d1 = func(train_l_d1, test_l_d1, 
                             y_train=y_train_68, model=p, d=q)
            ats_u_d0 = func(train_u_d0, test_u_d0, 
                             y_train=y_train_68, model=p, d=q)
            ats_u_d1 = func(train_u_d1, test_u_d1, 
                             y_train=y_train_68, model=p, d=q)
    
            Z = Z_train_68 + Z_test_68
            for i in Z:
                j = Z.index(i)
                features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_l_d0[j,:]
                features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
                features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d0[j,:]
                features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
                
    else:
        # we fix q=25 as a dummy variable
        q=25
        print(p,q)
        
        ats_l_d0 = func(train_l_d0, test_l_d0, 
                         y_train=y_train_68, model=p, d=q)
        ats_l_d1 = func(train_l_d1, test_l_d1, 
                         y_train=y_train_68, model=p, d=q)
        ats_u_d0 = func(train_u_d0, test_u_d0, 
                         y_train=y_train_68, model=p, d=q)
        ats_u_d1 = func(train_u_d1, test_u_d1, 
                         y_train=y_train_68, model=p, d=q)

        Z = Z_train_68 + Z_test_68
        for i in Z:
            j = Z.index(i)
            features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_l_d0[j,:]
            features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
            features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d0[j,:]
            features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
            
with open(path_feat + func.__name__ +'68_l_d0.pkl', 'wb') as f:
  pickle.dump(features_l_d0, f)
with open(path_feat + func.__name__ +'68_l_d1.pkl', 'wb') as f:
  pickle.dump(features_l_d1, f)
with open(path_feat + func.__name__ +'68_u_d0.pkl', 'wb') as f:
  pickle.dump(features_u_d0, f)
with open(path_feat + func.__name__ +'68_u_d1.pkl', 'wb') as f:
  pickle.dump(features_u_d1, f)

#%%

func = GetTemplateSystemFeature
    
print(func.__name__)
print(10)

train_l_d0 = []
train_l_d1 = []
train_u_d0 = []
train_u_d1 = []

for i in Z_train_10:
    train_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    train_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    train_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    train_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])
    
test_l_d0 = []
test_l_d1 = []
test_u_d0 = []
test_u_d1 = []

for i in Z_test_10:
    test_l_d0.append(pdiagrams["pdiag_l_d0_"+str(i)])
    test_l_d1.append(pdiagrams["pdiag_l_d1_"+str(i)])
    test_u_d0.append(pdiagrams["pdiag_u_d0_"+str(i)])
    test_u_d1.append(pdiagrams["pdiag_u_d1_"+str(i)])

    
features_l_d0 = dict()
features_l_d1 = dict()
features_u_d0 = dict()
features_u_d1 = dict()

for p in hyper_parameters[func.__name__][0]:
    if p=='gmm':
        for q in hyper_parameters[func.__name__][1]:
            print(p,q)
            
            ats_l_d0 = func(barcodes_train=train_l_d0, barcodes_test=test_l_d0, 
                             y_train=y_train_10, model=p, d=q)
            ats_l_d1 = func(barcodes_train=train_l_d1, barcodes_test=test_l_d1, 
                             y_train=y_train_10, model=p, d=q)
            ats_u_d0 = func(barcodes_train=train_u_d0, barcodes_test=test_u_d0, 
                             y_train=y_train_10, model=p, d=q)
            ats_u_d1 = func(barcodes_train=train_u_d1, barcodes_test=test_u_d1, 
                             y_train=y_train_10, model=p, d=q)
    
            Z = Z_train_10 + Z_test_10
            for i in Z:
                j = Z.index(i)
                features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_l_d0[j,:]
                features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
                features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d0[j,:]
                features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
                
    else:
        # we fix q=25 as a dummy variable
        q=25
        print(p,q)
        
        ats_l_d0 = func(train_l_d0, test_l_d0, 
                         y_train=y_train_10, model=p, d=q)
        ats_l_d1 = func(train_l_d1, test_l_d1, 
                         y_train=y_train_10, model=p, d=q)
        ats_u_d0 = func(train_u_d0, test_u_d0, 
                         y_train=y_train_10, model=p, d=q)
        ats_u_d1 = func(train_u_d1, test_u_d1, 
                         y_train=y_train_10, model=p, d=q)

        Z = Z_train_10 + Z_test_10
        for i in Z:
            j = Z.index(i)
            features_l_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_l_d0[j,:]
            features_l_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
            features_u_d0[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d0[j,:]
            features_u_d1[str(i)+'_'+str(p)+'_'+str(q)]=ats_u_d1[j,:]
            
with open(path_feat + func.__name__ +'10_l_d0.pkl', 'wb') as f:
  pickle.dump(features_l_d0, f)
with open(path_feat + func.__name__ +'10_l_d1.pkl', 'wb') as f:
  pickle.dump(features_l_d1, f)
with open(path_feat + func.__name__ +'10_u_d0.pkl', 'wb') as f:
  pickle.dump(features_u_d0, f)
with open(path_feat + func.__name__ +'10_u_d1.pkl', 'wb') as f:
  pickle.dump(features_u_d1, f)


#%%
print('DONE')


import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from vectorisation import *
import pandas as pd

#WARNING: the experiment do not work if the features are computed in a different
#computer than the one where the parameter are optimised and I don't know why.


data_path = 'Shrec14/data/'
feat_path = "Shrec14/features/"


#%%
# #Load the barcodes
# data = pd.read_csv(data_path+'Uli_data.csv')

# def reshapeVec(g):
#     A = np.array([g.dim,g.birth,g.death])
#     A = A.T
#     return A

# def getDgm(A, dim = 0):
#     if type(dim) != str:
#         if dim == 0:
#             A = A[np.where(np.logical_or(A[:,0] == dim, A[:,0] == -1))[0],1:]
            
#         if dim == 1:
#             A = A[np.where(np.logical_or(A[:,0] == dim, A[:,0] == -2))[0],1:]
    
#     return(A)

# dgmsDF = data.groupby(['freq', 'trial']).apply(reshapeVec)
# dgmsDF = dgmsDF.reset_index()
# dgmsDF = dgmsDF.rename(columns = {0:'CollectedDgm'})
# dgmsDF['Dgm1'] = dgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 1))

#%%
#Generate and save the labels

# def label(index):
#     if 0 <= index <= 19:
#         return 'male_neutral'
#     elif 20<= index <=39:
#         return 'male_bodybuilder'
#     elif 40<= index <=59:
#         return 'male_fat'
#     elif 60<= index <=79:
#         return 'male_thin'
#     elif 80<= index <=99:
#         return 'male_average'
#     elif 100<= index <=119:
#         return 'female_neutral'
#     elif 120<= index <=139:
#         return 'female_bodybuilder'
#     elif 140<= index <=159:
#         return 'female_fat'
#     elif 160<= index <=179:
#         return 'female_thin'
#     elif 180<= index <=199:
#         return 'female_average'
#     elif 200<= index <=219:
#         return 'child_neutral'
#     elif 220<= index <=239:
#         return 'child_bodybuilder'
#     elif 240<= index <=259:
#         return 'child_fat'
#     elif 260<= index <=279:
#         return 'child_thin'
#     elif 280<= index <=299:
#         return 'child_average'
#     else:
#         print('What are you giving me?')
        
# dgmsDF['TrainingLabel'] = dgmsDF.freq.apply(label)
# dgmsDF= dgmsDF.sample(frac=1)

# dgmsT = {}
# labelsD = {}
# label_names = {}
# for t in range(1,11):
#     T = dgmsDF[dgmsDF.trial==t]
#     dgmsT[str(t)] = np.array(T['Dgm1'])
    
#     labels = np.array(T['TrainingLabel'])
#     labels = pd.DataFrame(labels)
#     label_names = labels.copy()
    
#     label_unique = pd.DataFrame(labels)
#     label_unique = label_unique[0].unique()
    
#     i=0
#     for l in label_unique:
#         labels[labels == l]=i
#         i += 1
    
#     labelsD[str(t)] = labels[0].tolist()
#     label_names[str(t)] = label_names[0].tolist()
    
# with open(feat_path + 'dgmsT' +'.pkl', 'wb') as f:
#   pickle.dump(dgmsT, f)  
# with open(feat_path + 'labelsD' +'.pkl', 'wb') as f:
#   pickle.dump(labelsD, f)  
# with open(feat_path + 'labels_names' +'.pkl', 'wb') as f:
#   pickle.dump(label_names, f)
   
# Z_train = {}
# Z_test = {}
# y_train = {}
# y_test = {}
# for t in range(1,11):
#     Z_train[str(t)], Z_test[str(t)], y_train[str(t)], y_test[str(t)] = train_test_split(range(300), 
#                                                                         labelsD[str(t)], 
#                                                                         test_size=0.3, 
#                                                                         random_state=t)   
# with open(feat_path + 'Z_train' +'.pkl', 'wb') as f:
#   pickle.dump(Z_train, f)  
# with open(feat_path + 'Z_test' +'.pkl', 'wb') as f:
#   pickle.dump(Z_test, f)  
# with open(feat_path + 'y_train' +'.pkl', 'wb') as f:
#   pickle.dump(y_train, f)  
# with open(feat_path + 'y_test' +'.pkl', 'wb') as f:
#   pickle.dump(y_test, f)



#%%

with open(feat_path + 'dgmsT' +'.pkl', 'rb') as f:
    dgmsT = pickle.load(f)
with open(feat_path + 'Z_train' +'.pkl', 'rb') as f:
    Z_train = pickle.load(f)
with open(feat_path + 'Z_test' +'.pkl', 'rb') as f:
    Z_test = pickle.load(f)
with open(feat_path + 'y_train' +'.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(feat_path + 'y_test' +'.pkl', 'rb') as f:
    y_test = pickle.load(f)

#%%

hyper_parameters = {}
hyper_parameters['GetPersEntropyFeature'] = [50,100,200]
hyper_parameters['GetBettiCurveFeature'] = [50,100,200]
hyper_parameters['GetPersLifespanFeature'] = [50,100,200]
hyper_parameters['GetTopologicalVectorFeature'] = [5, 10, 20]
hyper_parameters['GetPersImageFeature'] = [10,25,50,100,200]
hyper_parameters['GetAtolFeature'] = [2,4,8,16]
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
    features={}
    print(func.__name__)
    for t in range(1,11):
        Z = Z_train[str(t)]+Z_test[str(t)]
        dgms = dgmsT[str(t)]
        for i in Z:
            features[str(t)+'_'+str(i)]=func(dgms[i]) 
        
    with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
      pickle.dump(features, f)
      
#%%
#Methods with only one parameter
func_list = [
              # GetPersEntropyFeature,
              # GetBettiCurveFeature,
              # GetTopologicalVectorFeature,
              # GetPersLifespanFeature,
              # GetPersImageFeature,
            ]


for func in func_list:
    features={}
    print(func.__name__)
    for p in hyper_parameters[func.__name__]:
        print(str(p))
        for t in range(1,11):
            Z = Z_train[str(t)]+Z_test[str(t)]
            dgms = dgmsT[str(t)]
            for i in Z:
                features[str(t)+'_'+str(p)+'_'+str(i)]=func(dgms[i],p) 
            
        with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
          pickle.dump(features, f)
          
#%%
func_list = [
              GetAtolFeature
            ]

from vectorisation.bar_cleaner import bar_cleaner

for func in func_list:
    features={}
    print(func.__name__)
    for p in hyper_parameters[func.__name__]:
        for t in range(1,10):
            print(p,t)
            Z = Z_train[str(t)]+Z_test[str(t)]
            dgms = dgmsT[str(t)]
            atol_list = func(dgms,p) 
            for i in Z:
                features[str(t)+'_'+str(p)+'_'+str(i)]=atol_list[i,:]
            
        with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
          pickle.dump(features, f)
          

#%%
#Methods with two parameter
func_list = [
             # GetPersSilhouetteFeature,
             # GetComplexPolynomialFeature,
             # GetPersLandscapeFeature
            ]

for func in func_list:
    features={}
    print(func.__name__)
    for p in hyper_parameters[func.__name__][0]:
        for q in hyper_parameters[func.__name__][1]:
            print(str(p),str(q))
            for t in range(1,11):
                Z = Z_train[str(t)]+Z_test[str(t)]
                dgms = dgmsT[str(t)]
                for i in Z:
                    features[str(t)+'_'+str(p)+'_'+str(q)+'_'+str(i)]=func(dgms[i],p,q) 
                
            with open(feat_path + func.__name__ +'.pkl', 'wb') as f:
              pickle.dump(features, f)



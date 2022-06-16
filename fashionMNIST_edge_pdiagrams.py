#import pickle
import numpy as np
from skimage import feature
from scipy.ndimage import distance_transform_bf
from fashion_mnist import mnist_reader
from vectorisation import *

path_feat = "fashion_mnist/features/"
path_dia = "fashion_mnist/pdiagrams/"

n_train = 60000
n_total = 70000

#%%
images_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
images_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')
indexes_train = range(n_train)
indexes_test = range(n_train, n_total)
images = np.vstack([images_train, images_test])
labels = np.array(y_train.tolist() + y_test.tolist())

#%%
to2828 = lambda ima : ima.reshape([28,28])
edger = lambda ima : feature.canny(image=ima, low_threshold=20, high_threshold=170)
filt_taxi = lambda ima : distance_transform_bf(ima, metric='taxicab')
filt_euc = lambda ima : distance_transform_bf(ima)**2

taxi_pipeline = lambda ima : filt_taxi(edger(to2828(ima)))
euc_pipeline = lambda ima : filt_euc(edger(to2828(ima)))

taxi_complexes = np.array(list(map(taxi_pipeline, images)))
euc_complexes = np.array(list(map(euc_pipeline, images)))

#%%

for i in range(n_train):
    print(i)
    dgms = GetCubicalComplexPDs(img=taxi_complexes[i].reshape(784,), img_dim=[28,28])
    np.savetxt(path_dia + "taxi_d0_"+str(i),dgms[0])
    np.savetxt(path_dia + "taxi_d1_"+str(i),dgms[1])
    
    dgms = GetCubicalComplexPDs(img=euc_complexes[i].reshape(784,), img_dim=[28,28])
    np.savetxt(path_dia + "euc_d0_"+str(i),dgms[0])
    np.savetxt(path_dia + "euc_d1_"+str(i),dgms[1])
    
for i in range(n_train, n_total):
    print(i)
    dgms = GetCubicalComplexPDs(img=taxi_complexes[i].reshape(784,), img_dim=[28,28])
    np.savetxt(path_dia + "taxi_d0_"+str(i),dgms[0])
    np.savetxt(path_dia + "taxi_d1_"+str(i),dgms[1])
    
    dgms = GetCubicalComplexPDs(img=euc_complexes[i].reshape(784,), img_dim=[28,28])
    np.savetxt(path_dia + "euc_d0_"+str(i),dgms[0])
    np.savetxt(path_dia + "euc_d1_"+str(i),dgms[1])
    
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
    pdiagrams["pdiag_taxi_d0_"+str(i)]= safe_load(path_diag + "taxi_d0_"+str(i))
    pdiagrams["pdiag_taxi_d1_"+str(i)]= safe_load(path_diag + "taxi_d1_"+str(i))
    pdiagrams["pdiag_euc_d0_"+str(i)]= safe_load(path_diag + "euc_d0_"+str(i))
    pdiagrams["pdiag_euc_d1_"+str(i)]= safe_load(path_diag + "euc_d1_"+str(i))

#%%
    
func_list = [GetPersStats,
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

features_taxi_d0 = dict()
features_taxi_d1 = dict()
features_euc_d0 = dict()
features_euc_d1 = dict()

for func in func_list:
    print(func.__name__)
    for i in range(n_total):
        features_taxi_d0[(func.__name__)+'_'+str(i)]=func(pdiagrams["pdiag_taxi_d0_"+str(i)])
        features_taxi_d1[(func.__name__)+'_'+str(i)]=func(pdiagrams["pdiag_taxi_d1_"+str(i)])
        features_euc_d0[(func.__name__)+'_'+str(i)]=func(pdiagrams["pdiag_euc_d0_"+str(i)])
        features_euc_d1[(func.__name__)+'_'+str(i)]=func(pdiagrams["pdiag_euc_d1_"+str(i)])

#%%
        
func = GetComplexPolynomialFeature
complex_coeff = []

for i in range(n_total):
    complex_coeff.append(features_taxi_d0[(func.__name__)+'_'+str(i)])
for i in range(n_total):
    complex_coeff.append(features_taxi_d1[(func.__name__)+'_'+str(i)])
for i in range(n_total):
    complex_coeff.append(features_euc_d0[(func.__name__)+'_'+str(i)])
for i in range(n_total):
    complex_coeff.append(features_euc_d1[(func.__name__)+'_'+str(i)])

complex_coeff = float64to32(complex_coeff)

for i in range(n_total):
    features_taxi_d0[(func.__name__)+'_'+str(i)]=complex_coeff[i]
    features_taxi_d1[(func.__name__)+'_'+str(i)]=complex_coeff[n_total+i]  
    features_euc_d0[(func.__name__)+'_'+str(i)]=complex_coeff[2*n_total+i]   
    features_euc_d1[(func.__name__)+'_'+str(i)]=complex_coeff[3*n_total+i]
    

#%%

with open(path_feat + 'taxi_d0.pkl', 'wb') as f:
  pickle.dump(features_taxi_d0, f)
with open(path_feat+'taxi_d1.pkl', 'wb') as f:
  pickle.dump(features_taxi_d1, f)
with open(path_feat + 'euc_d0.pkl', 'wb') as f:
  pickle.dump(features_euc_d0, f)
with open(path_feat + 'euc_d1.pkl', 'wb') as f:
  pickle.dump(features_euc_d1, f)
    
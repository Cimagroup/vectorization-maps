from feature_computation import *
import vectorisation as vect

vec_methods = dict()
#vec_methods['GetPersStats']=(),
#vec_methods['GetCarlssonCoordinatesFeature']=(),
vec_methods['GetPersEntropyFeature'] = [[15,30,50]]
#vec_methods['GetBettiCurveFeature'] = [[15,30,50]]
#vec_methods['GetPersLifespanFeature'] = [[15,30,50]]
#vec_methods['GetTopologicalVectorFeature'] = [[3, 5, 10]]
#vec_methods['GetAtolFeature'] = [[2,4,8,16]]
#vec_methods['GetPersImageFeature'] = [[3,6,12,20]]
vec_methods['GetPersSilhouetteFeature'] = [[15,30,50], [0,1,2,5]]
#vec_methods['GetComplexPolynomialFeature'] = [[3, 5, 10],['R', 'S', 'T']]
#vec_methods['GetPersLandscapeFeature'] = [[15,30,50], [1,2,3,5]]
vec_methods['GetTemplateFunctionFeature'] = [[1,2,3,5], [.5, 1, 2]]
#vec_methods['GetAdaptativeSystemFeature'] = [['gmm'], 
#                                                [1,2,3,4,5,10,15]]

# LOADING DIAGRAMS
from fashion_mnist import mnist_reader
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
    #pdiagrams["pdiag_taxi_l_"+str(i)]= safe_load(path_diag + "taxi_l_"+str(i))
    pdiagrams["pdiag_taxi_u_"+str(i)]= safe_load(path_diag + "taxi_u_"+str(i))

    
# Feature Computation    
feature_computation(vec_methods,pdiagrams,"pdiag_taxi_u_",n_train,n_total)
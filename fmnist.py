from sklearn.model_selection import GridSearchCV
from feature_computation import *
import vectorisation as vect
from auxiliary_functions import *
from numpy.random import seed
import pickle

s=1
seed(s)

normalization=False
n_iters = 100

# In[2]:

from fashion_mnist import mnist_reader
path_feat = "fashion_mnist/features/"
path_diag= "fashion_mnist/pdiagrams/"
path_results = "results/"

# In[3]:

vec_parameters = dict()
vec_parameters['GetPersStats']=(),
vec_parameters['GetCarlssonCoordinatesFeature']=(),
vec_parameters['GetPersEntropyFeature'] = [[15,30,50]]
vec_parameters['GetBettiCurveFeature'] = [[15,30,50]]
vec_parameters['GetPersLifespanFeature'] = [[15,30,50]]
vec_parameters['GetTopologicalVectorFeature'] = [[3, 5, 10]]
vec_parameters['GetAtolFeature'] = [[2,4,8,16]]
vec_parameters['GetPersImageFeature'] = [[0.05,0.5,1],[3,6,12,20]]
vec_parameters['GetPersSilhouetteFeature'] = [[15,30,50], [0,1,2,5]]
vec_parameters['GetComplexPolynomialFeature'] = [[3, 5, 10],['R', 'S', 'T']]
vec_parameters['GetPersLandscapeFeature'] = [[15,30,50], [1,2,3,5]]
vec_parameters['GetTemplateFunctionFeature'] = [[2,3,5,10], [.5, 1, 2]]
vec_parameters['GetPersTropicalCoordinatesFeature'] = [[10,50,250]]
vec_parameters['GetAdaptativeSystemFeature'] = [['gmm'], 
                                                [3,4,5,10,15]]


# In[4]:


from parameter_optimization import *

onlyForest = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100]},
 ]

searchG = GridSearchCV(
    main_classifier(), param_grid=onlyForest, cv=5,
    return_train_score=True, scoring='accuracy'
)

# In[5]:

_, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
_, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

train_index = list(range(len(y_train)))
test_index = list(range(len(y_train), len(y_train)+len(y_test)))
index = train_index+test_index

# In[6]:

pdiagrams = dict()

for i in index:
    #pdiagrams["pdiag_taxi_l_"+str(i)]= safe_load(path_diag + "taxi_l_"+str(i))
    pdiagrams["pdiag_taxi_u_"+str(i)]= safe_load(path_diag + "taxi_u_"+str(i))


# In[7]:

func_list = [getattr(vect, keys) for keys in vec_parameters.keys()]
for func in func_list:
    feature_dictionary = dict()
    vec_methods = dict()
    vec_methods[func.__name__] = vec_parameters[func.__name__]
    
    feature_dictionary=feature_computation(vec_methods,pdiagrams,"pdiag_taxi_u_",
                                           train_index, test_index)

    with open(path_results+'FMNIST_feature_'+func.__name__+'.pkl', 'wb') as f:
        pickle.dump(feature_dictionary, f)
        
    best_scores=parameter_optimization(train_index, y_train, vec_methods, 
                                       feature_dictionary, searchG, normalization)
    
    print("Parameter optimization:",best_scores)
    
    with open(path_results+'FMNIST_best_scores_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(best_scores, f)

    train_scores, test_scores = scores(train_index, y_train, test_index, y_test, 
                                       vec_methods, feature_dictionary, best_scores,  
                                       n_iters, normalization)

    print("The train accuracy is", train_scores)
    print("The test accuracy is", test_scores)
    
    with open(path_results+'FMNIST_train_scores_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(train_scores, f)
    with open(path_results+'FMNIST_test_scores_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(test_scores, f)


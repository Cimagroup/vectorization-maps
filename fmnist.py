from sklearn.model_selection import GridSearchCV
from feature_computation import *
import vectorisation as vect
from auxiliary_functions import *
from numpy.random import seed
import pickle

s=1
seed(s)


# In[2]:


vec_methods = dict()
#vec_methods['GetPersStats']=(),
#vec_methods['GetCarlssonCoordinatesFeature']=(),
#vec_methods['GetPersEntropyFeature'] = [[15,30,50]]
#vec_methods['GetBettiCurveFeature'] = [[15,30,50]]
#vec_methods['GetPersLifespanFeature'] = [[15,30,50]]
#vec_methods['GetTopologicalVectorFeature'] = [[3, 5, 10]]
vec_methods['GetAtolFeature'] = [[16]]#[[2,4,8,16]]
#vec_methods['GetPersImageFeature'] = [[3,6,12,20]]
#vec_methods['GetPersSilhouetteFeature'] = [[15,30,50], [0,1,2,5]]
#vec_methods['GetComplexPolynomialFeature'] = [[3, 5, 10],['R', 'S', 'T']]
#vec_methods['GetPersLandscapeFeature'] = [[15,30,50], [1,2,3,5]]
#vec_methods['GetTemplateFunctionFeature'] = [[1,2,3,5], [.5, 1, 2]]
#vec_methods['GetAdaptativeSystemFeature'] = [['gmm'], 
#                                                [1,2,3,4,5,10,15]]


# In[3]:

from fashion_mnist import mnist_reader
path_feat = "fashion_mnist/features/"
path_diag= "fashion_mnist/pdiagrams/"
path_results = "results/"

# In[ ]:

_, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
_, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

train_index = list(range(len(y_train)))
test_index = list(range(len(y_train), len(y_train)+len(y_test)))
index = train_index+test_index

# In[ ]:

pdiagrams = dict()

for i in index:
    #pdiagrams["pdiag_taxi_l_"+str(i)]= safe_load(path_diag + "taxi_l_"+str(i))
    pdiagrams["pdiag_taxi_u_"+str(i)]= safe_load(path_diag + "taxi_u_"+str(i))


# In[ ]:


feature_dictionary=feature_computation(vec_methods,pdiagrams,"pdiag_taxi_u_",
                                       train_index, test_index)

with open(path_results+'FMNIST_feature_dictionary.pkl', 'wb') as f:
  pickle.dump(feature_dictionary, f)

# In[ ]:


from parameter_optimization import *

onlyForest = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100]},
 ]

searchG = GridSearchCV(
    main_classifier(), param_grid=onlyForest, cv=5,
    return_train_score=True, scoring='accuracy'
)


# In[ ]:


best_scores=parameter_optimization(train_index, y_train, vec_methods, 
                                   feature_dictionary, searchG, normalization=False)


# In[ ]:


print("Parameter optimization:",best_scores)
with open(path_results+'FMNIST_best_scores.pkl', 'wb') as f:
  pickle.dump(best_scores, f)


# In[ ]:

# ## Classification

n_iters = 100
train_scores, test_scores = scores(train_index, y_train, test_index, y_test, 
                                   vec_methods, feature_dictionary, best_scores,  
                                   n_iters, normalization=True)

# In[ ]:

print("The train accuracy is", train_scores)
print("The test accuracy is", test_scores)

with open(path_results+'FMNIST_train_scores.pkl', 'wb') as f:
  pickle.dump(train_scores, f)
with open(path_results+'FMNIST_test_scores.pkl', 'wb') as f:
  pickle.dump(test_scores, f)


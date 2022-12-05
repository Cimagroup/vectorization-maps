import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import vectorisation as vect
import numpy as np
from direct_optimisation import main_classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from types import SimpleNamespace
from numpy.random import seed
#Barcodes with just one bar are loaded as a 1d-array.
#We force them to be a 2d-array

def safe_load(x):
    pd = np.loadtxt(x)
    if (len(pd.shape)==1) and (pd.shape[0]>0): 
        pd = pd.reshape(1,2)
    return pd

def load_parameters(func,d):
    func_parameters =d[func.__name__]
    func_parameters = list(itertools.product(*func_parameters))
    if func_parameters == []:
        l = 0
        func_parameters = [[1]]
    else:
        l = np.shape(func_parameters[0])[0]
    if l == 1:
        func_parameters = [[x[0]] for x in func_parameters]
    elif l == 2:
        func_parameters = [[x[0],x[1]] for x in func_parameters]
    return func_parameters
    
def best_parameter(features, func):
    keys = [k for k in features.keys() if func in k]
    index = np.argmax([features[key][1] for key in keys])
    keys_l=list(keys)
    best_params_key = keys_l[index]
    return best_params_key

def build_dataset_from_features(train_index,y_train,func,feature_dictionary,parameter):     
    X_train = [feature_dictionary[func.__name__+'_'+str(parameter)][str(i)] for i in train_index]
    return (np.array(X_train),y_train)

def safe_load(x):
    pd = np.loadtxt(x)
    if (len(pd.shape)==1) and (pd.shape[0]>0): 
        pd = pd.reshape(1,2)
    return pd

def scores(train_index, y_train, test_index, y_test, vectorization_methods, 
           feature_dictionary, best_scores, n_iters, normalization):
    func_list = [getattr(vect, keys) for keys in vectorization_methods.keys()]
    train_scores = {}
    test_scores = {}
    for func in func_list:
        train_scores[func.__name__]=[]
        test_scores[func.__name__]=[]
    for i in range(n_iters):
        #best_parameter = deepcopy(best_scores)
        aux1, aux2 = classification(train_index, y_train, test_index, y_test, vectorization_methods, 
                                    feature_dictionary, best_scores, normalization)
        for func in func_list:
            train_scores[func.__name__].append(aux1[func.__name__])
            test_scores[func.__name__].append(aux2[func.__name__])

    for func in func_list:
        train_scores[func.__name__] = (np.mean(train_scores[func.__name__]), np.std(train_scores[func.__name__]))
        test_scores[func.__name__] = (np.mean(test_scores[func.__name__]), np.std(test_scores[func.__name__]))
        
    return train_scores, test_scores

def classification(train_index, y_train, test_index,y_test, vectorisation_methods, 
                   feature_vectors, best_scores, normalization): 
    # Initial parameters
    base_estimator='RF' 
    n_estimators=100
    C=1.0
    kernel='rbf'
    gamma=0.1
    degree=3
    func_list = vectorisation_methods.keys()
    train_scores = dict()
    test_scores = dict()
    for func in func_list:
        best_params_key = best_parameter(best_scores,func)
        classifier_parameters = best_scores[best_params_key][0]
        # Update the parameters depending on best_scores from the parameter
        # optimization process.
        if classifier_parameters['base_estimator']=='RF':            
            n_estimators=classifier_parameters['n_estimators']
        else:
            base_estimator=classifier_parameters['base_estimator']
            C=classifier_parameters['C']
            kernel=classifier_parameters['kernel']
            gamma=classifier_parameters['gamma']
        # for key,val in classifier_parameters.items():
        #     #exec(key + '=val')
        #     ps.append(val)
        method = main_classifier(base_estimator,n_estimators, C, kernel, gamma, degree)

        X_train = [feature_vectors[best_params_key][str(i)] for i in train_index]
        X_test = [feature_vectors[best_params_key][str(i)] for i in test_index]
        X_train, X_test = np.array(X_train), np.array(X_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
        if normalization:
            mm_scaler = MinMaxScaler()
            X_train = mm_scaler.fit_transform(X_train)
            X_test = mm_scaler.transform(X_test)
        method.fit(X_train, y_train)      
        train_scores[func]=np.mean(y_train.ravel() == method.predict(X_train))
        test_scores[func]=np.mean(y_test.ravel() == method.predict(X_test))
        #train_scores[func] = method.score(X_train, y_train)
        #test_scores[func] = method.score(X_test, y_test)   
    return train_scores, test_scores
import vectorisation as vect
from auxiliary_functions import *
from copy import deepcopy
 
#%%
def feature_computation(vectorisation_methods,pdiagrams,diag_key,train_index,test_index):  
 
    func_list = [getattr(vect, keys) for keys in vectorisation_methods.keys()]
    features = dict()
    index = train_index + test_index
    for func in func_list:
        func_parameters = load_parameters(func,vectorisation_methods)
        for p in func_parameters:
            features_func = dict()
            if func not in  [vect.GetAtolFeature,vect.GetTemplateFunctionFeature, vect.GetAdaptativeSystemFeature]:
                for i in index:
                    barcode = pdiagrams[diag_key+str(i)]
                    features_func[str(i)]=func(barcode,*p)
                features[func.__name__+'_'+str(p)] = features_func
            if func == vect.GetAtolFeature:
                features_list = func([pdiagrams[diag_key+str(i)] for i in index],*p)
                for i in index:
                    j = index.index(i)
                    features_func[str(i)]= features_list[j,:]
                features[func.__name__+'_'+str(p)]= features_func
            if func in [vect.GetTemplateFunctionFeature, vect.GetAdaptativeSystemFeature]:
                train = [pdiagrams[diag_key+str(i)] for i in train_index]
                test  = [pdiagrams[diag_key+str(i)] for i in test_index]
                features_list = func(train,test,*p)
                for i in index:
                    j = index.index(i)
                    features_func[str(i)]= features_list[j,:]
                features[func.__name__+'_'+str(p)]= features_func
    return features 
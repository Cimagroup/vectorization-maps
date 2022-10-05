from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from vectorisation import GetPersTropicalCoordinatesFeature as T
import numpy as np

class tropical_classifier(BaseEstimator):

    def __init__(self, base_estimator='RF', n_estimators=100, C=1.0, 
                 kernel='rbf', gamma=0.1, degree=3, r=28):
    
        self._estimator_type = "classifier"
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.r = r
        
    
    def fit(self, X, y):
        if self.base_estimator=='RF':
            self.estimator_= RandomForestClassifier(self.n_estimators)
        elif self.base_estimator=='SVM':
            self.estimator_=SVC(C=self.C, kernel = self.kernel, 
                                     gamma=self.gamma, degree=self.degree)   
        else :
            print('The estimator must be "RF" or "SVM"')
        
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
        

        path_diag = "Outex-TC-00024/pdiagrams/"
        new_X = []
        for i in X:
            T1 = T(safe_load(path_diag+"l_d0_"+str(i)), r=self.r)
            T2 = T(safe_load(path_diag+"l_d1_"+str(i)), r=self.r)
            T3 = T(safe_load(path_diag+"u_d0_"+str(i)), r=self.r)
            T4 = T(safe_load(path_diag+"u_d1_"+str(i)), r=self.r)
            new_X.append(np.hstack((T1,T2,T3,T4)))
        
        new_X=np.array(new_X)        
        self.X_ = new_X
        self.y_ = y
        
        return self.estimator_.fit(self.X_, self.y_)
    
    def predict(self, X):
        
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
        

        path_diag = "Outex-TC-00024/pdiagrams/"
        new_X = []
        for i in X:
            T1 = T(safe_load(path_diag+"l_d0_"+str(i)), r=self.r)
            T2 = T(safe_load(path_diag+"l_d1_"+str(i)), r=self.r)
            T3 = T(safe_load(path_diag+"u_d0_"+str(i)), r=self.r)
            T4 = T(safe_load(path_diag+"u_d1_"+str(i)), r=self.r)
            new_X.append(np.hstack((T1,T2,T3,T4)))
        
        np.array(new_X)        
        
        return self.estimator_.predict(new_X)

    def classes_(self):
        if self.estimator_:
            return self.base_estimator_.classes_
    
    
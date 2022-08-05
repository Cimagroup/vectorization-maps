from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class main_classifier(BaseEstimator):

    def __init__(self, base_estimator='RF', n_estimators=100, C=1.0, 
                 kernel='rbf', gamma=0.1, degree=3):
    
        self._estimator_type = "classifier"
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        
    
    def fit(self, X, y):
        if self.base_estimator=='RF':
            self.estimator_=RandomForestClassifier(self.n_estimators)
        elif self.base_estimator=='SVM':
            self.estimator_=SVC(C=self.C, kernel = self.kernel, 
                                     gamma=self.gamma, degree=self.degree)   
        else :
            print('The estimator must be "RF" or "SVM"')
        
        self.X_ = X
        self.y_ = y
        
        return self.estimator_.fit(self.X_, self.y_)
    
    def predict(self, X):
        return self.estimator_.predict(X)

    def classes_(self):
        if self.estimator_:
            return self.base_estimator_.classes_
    
    
import pywavefront
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



def classification(path,labels):
# path to computed features saved as dictionary
# labels of the dataset
# Return a dictionary with the performance.
    d = dict()
    for i in range(0,400):
        with open(path+str(i)+".pkl", 'rb') as f:
            d[str(i)] = pickle.load(f)
    scores = dict()
    for fun in d["0"].keys():
        X=[d[str(i)][fun] for i in range(400)]
        clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
        score = cross_val_score(clf, X, labels, cv=5)
        scores[fun]=score.mean()
    with open('saved_scores.pkl', 'wb') as f:
      pickle.dump(scores, f)
import numpy as np
from gudhi import representations
from sklearn.cluster import KMeans
quantiser = KMeans(n_clusters=2, random_state=202006)
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetAtolFeature"]

def GetAtolFeature(barcode, qt=quantiser):
    feature_vector = []
    barcode = bar_cleaner(barcode)
    
    if(np.size(barcode) > 0):
        atol = representations.vector_methods.Atol(quantiser=qt)
        feature_vector = atol.fit_transform([barcode])[0]

    return feature_vector
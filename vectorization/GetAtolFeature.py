import numpy as np
from gudhi import representations

__all__ = ["GetAtolFeature"]

def GetAtolFeature(barcode, qt):
    feature_vector = []

    if(np.size(barcode) > 0):
        atol = representations.vector_methods.Atol(quantiser=qt)
        feature_vector = atol.fit_transform([barcode])[0]

    return feature_vector
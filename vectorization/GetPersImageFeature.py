import numpy as np
from gudhi import representations

__all__ = ["GetPersImageFeature"]

def GetPersImageFeature(barcode, res=[6,6]):

    if(np.size(barcode) > 0):
        perImg = representations.PersistenceImage(resolution=res)
        feature_vector = perImg.fit_transform([barcode])[0]
    else:
        feature_vector = np.zeros(res[0]*res[1])

    return feature_vector

import numpy as np
from gudhi import representations

__all__ = ["GetPersImageFeature"]

def GetPersImageFeature(barcode, r=48):
    res=[r,r]
    if(np.size(barcode) > 0):
        perImg = representations.PersistenceImage(resolution=res)
        feature_vector = perImg.fit_transform([barcode])[0]
    else:
        feature_vector = np.zeros(res[0]**2)

    return feature_vector

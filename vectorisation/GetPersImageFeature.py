import numpy as np
from gudhi import representations
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetPersImageFeature"]

def GetPersImageFeature(barcode, bw=0.2, r=20, ):
    barcode = bar_cleaner(barcode)
    res=[r,r]
    if(np.size(barcode) > 0):
        perImg = representations.PersistenceImage(bandwidth=bw, resolution=res)
        feature_vector = perImg.fit_transform([barcode])[0]
    else:
        feature_vector = np.zeros(res[0]**2)

    return feature_vector

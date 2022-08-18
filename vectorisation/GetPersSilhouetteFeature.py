import numpy as np
from gudhi import representations

__all__ = ["GetPersSilhouetteFeature"]

def GetPersSilhouetteFeature(barcode, res=100, w=1):

    if(np.size(barcode) > 0):
        persSilhouette = representations.vector_methods.Silhouette(resolution=res, weight=lambda x : w)
        feature_vector = persSilhouette.fit_transform([barcode])[0]
    else:
    	feature_vector = np.zeros(res)

    return feature_vector

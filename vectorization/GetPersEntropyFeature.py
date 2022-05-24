import numpy as np
import persistence_curves as pc

__all__ = ["GetPersEntropyFeature"]

def GetPersEntropyFeature(barcode, res=100):

    if(np.size(barcode) > 0):
        ent = pc.Entropy(mode='vector', resolution = res)
        feature_vector = ent.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
        
    return feature_vector
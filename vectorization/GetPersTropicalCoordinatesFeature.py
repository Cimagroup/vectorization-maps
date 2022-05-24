import numpy as np
from copy import deepcopy

__all__ = ["GetPersTropicalCoordinatesFeature"]

def GetPersTropicalCoordinatesFeature(barcode, r=28):
    
    if(np.size(barcode) > 0):
        #change the deaths by the lifetime
        new_barcode = deepcopy(barcode)
        new_barcode[:,1] = new_barcode[:,1]-new_barcode[:,0]
        #sort them so the bars with the longest lifetime appears first
        new_barcode = new_barcode[np.argsort(-new_barcode[:,1])]
        #Write the output of the selected polynomials
        pol_max1 = new_barcode[0,1]
        pol_max2 = new_barcode[0,1] + new_barcode[1,1]
        pol_max3 = new_barcode[0,1] + new_barcode[1,1] + new_barcode[2,1]
        pol_max4 = new_barcode[0,1] + new_barcode[1,1] + new_barcode[2,1] + new_barcode[3,1]
        total_length = sum(new_barcode[:,1])
        #In each row, take the minimum between the birth time and r*lifetime
        aux_array = np.array(list(map(lambda x : min(r*x[1], x[0]), new_barcode)))
        pol_r = sum(aux_array)
        M = max(aux_array + new_barcode[:,1])
        pol_r2 = sum(M - (aux_array + new_barcode[:,1]))
        
        feature_vector = np.array([pol_max1, pol_max2, pol_max3, pol_max4,
                                   total_length, pol_r, pol_r2])
    else:
    	feature_vector = np.zeros(7)
            
    return feature_vector
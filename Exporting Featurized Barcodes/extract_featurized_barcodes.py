# # Featurized PH Barcodes
# This file contains the implementations of all featurized PH barcods

# Import dependencies
import numpy as np
from gudhi import CubicalComplex
import teaspoon.ML.feature_functions as Ff
import persistence_curves as pc


# 1. Function to compute PH binnings
def GetPersBinning(barcode, thresholds):
    # This function takes a ripser dim n matrix and a threshold and compute 
    # the intersection of the thresholds with each life line.
    
    n = np.size(barcode, 0)
    binning = []
    
    if(n > 0):
        if(np.size(thresholds) == 1):
            thresholds = [thresholds]
            
        for threshold in thresholds:
            int_count = 0
            for i in range(n):
                if(threshold >= barcode[i,0] and threshold <= barcode[i,1]):
                    int_count += 1
                    
            binning.append(int_count)
        
    return binning

# 2. Function to compute PH statistics
def GetPersStats(barcode):
    # Computing Statistics from Persistent Barcodes

    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        diff_barcode = np.subtract([i[1] for i in barcode], [i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars        
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Number of Bars
        bc_count = len(diff_barcode)

        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                     bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_count])
    else:
        bar_stats= np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    bar_stats[~np.isfinite(bar_stats)] = 0
    
    return bar_stats

# 3. Function to compute Persistence Image
def GetPersImageFeature(barcode, persIm):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persIm.fit_transform([barcode])
    
    return feature_vectors

# 4. Function to compute Persistence Landscape
def GetPersLandscapeFeature(barcode, persLand):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persLand.fit_transform([barcode])
    
    return feature_vectors

# 5. Function to compute Persistence Entropy
def GetPersEntropyFeature(barcode):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        ent = pc.Entropy(mode='vector')
        feature_vectors = ent.fit_transform(barcode)
        
    return feature_vectors

# 6. Function to compute Betti Curve
def GetBettiCurveFeature(barcode, bettiCurve):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = bettiCurve.fit_transform([barcode])
    
    return feature_vectors

# 7. Function to compute Carlsson Coordinates
def GetCarlssonCoordinatesFeature(barcode, FN=3):
    if(np.size(barcode) > 0):
        featureMatrix, _, _ = Ff.F_CCoordinates(barcode,FN)
        return featureMatrix
    else:
        return [],[],[]

# 8. Function to compute Persistence Codebooks (incomplete)
# def GetPersistenceCodebooksFeature(barcode, pbow, wpbow, spbow):
#     if(np.size(barcode) > 0):
#         pbow_diagrams  = pbow.transform(barcode)
#         wpbow_diagrams = wpbow.transform(barcode)
#         spbow_diagrams = spbow.transform(barcode)
#         return pbow_diagrams,wpbow_diagrams,spbow_diagrams
#     else:
#         return [],[],[]

# 9. Function to compute Persistence Silhouette
def GetPersSilhouetteFeature(barcode, persSilhouette):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persSilhouette.fit_transform([barcode])
    
    return feature_vectors

# 10. Function to compute Topological Vector
def GetTopologicalVectorFeature(barcode, topologicalVector):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = topologicalVector.fit_transform([barcode])
    
    return feature_vectors

# 11. Function to compute Atol
def GetAtolFeature(barcode, atol):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = atol.fit_transform([barcode])
    
    return feature_vectors

# 12. Function to compute Complex Polynomial
def GetComplexPolynomialFeature(barcode, complexPolynomial):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = complexPolynomial.fit_transform([barcode])
    
    return feature_vectors

# Function to compute pds
def GetCubicalComplexPDs(img, img_dim):
    cub_filtration = CubicalComplex(dimensions = img_dim, top_dimensional_cells = img)
    cub_filtration.persistence()
    pds = [cub_filtration.persistence_intervals_in_dimension(0),
           cub_filtration.persistence_intervals_in_dimension(1)]
    
    pds[0] = pds[0][pds[0][:, 0] != np.inf]
    pds[0] = pds[0][pds[0][:, 1] != np.inf]
    pds[1] = pds[1][pds[1][:, 0] != np.inf]
    pds[1] = pds[1][pds[1][:, 1] != np.inf]
    
    return pds
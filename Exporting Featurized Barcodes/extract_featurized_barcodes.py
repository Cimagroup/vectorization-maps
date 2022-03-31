# # Featurized PH Barcodes
# This file contains the implementations of all featurized PH barcods

# Import dependencies
import numpy as np
from gudhi import CubicalComplex, representations
import teaspoon.ML.feature_functions as Ff
import persistence_curves as pc


# 1. Function to compute PH binnings
#def GetPersBinning(barcode, thresholds):
    # This function takes a ripser dim n matrix and a threshold and compute 
    # the intersection of the thresholds with each life line.
    
#    n = np.size(barcode, 0)
#    binning = []
    
#    if(n > 0):
#        if(np.size(thresholds) == 1):
#            thresholds = [thresholds]
            
#        for threshold in thresholds:
#            int_count = 0
#            for i in range(n):
#                if(threshold >= barcode[i,0] and threshold <= barcode[i,1]):
#                    int_count += 1
                    
#            binning.append(int_count)
        
#    return binning

# 1. Function to compute PH statistics
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
        # Persitent Entropy
        ent = pc.Entropy()
        bc_ent = ent.fit_transform([barcode])

        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                     bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_count,#])
                     bc_ent[0][0]])
    else:
        bar_stats= np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    bar_stats[~np.isfinite(bar_stats)] = 0
    
    return bar_stats

# 2. Function to compute Persistence Image
def GetPersImageFeature(barcode, res):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        perImg = representations.PersistenceImage(resolution=res)
        feature_vector = perImg.fit_transform([barcode])[0]
    
    return feature_vector

# 3. Function to compute Persistence Landscape
def GetPersLandscapeFeature(barcode, res):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        perLand = representations.Landscape(resolution=res)
        feature_vector = perLand.fit_transform([barcode])[0]
    
    return feature_vector

# 4. Function to compute Persistence Entropy
def GetPersEntropyFeature(barcode):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        ent = pc.Entropy(mode='vector')
        feature_vector = ent.fit_transform(barcode).flatten()
        
    return feature_vector

# 5. Function to compute Betti Curve
def GetBettiCurveFeature(barcode, res):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        bettiCurve = representations.vector_methods.BettiCurve(resolution=res)
        feature_vector = bettiCurve.fit_transform([barcode])[0]
    
    return feature_vector

# 6. Function to compute Carlsson Coordinates
def GetCarlssonCoordinatesFeature(barcode, FN=3):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        featureMatrix, _, _ = Ff.F_CCoordinates(barcode,FN)
        feature_vector = np.concatenate([mat.flatten() for mat in featureMatrix])
        
    return feature_vector

# 7. Function to compute Persistence Codebooks (incomplete)
# def GetPersistenceCodebooksFeature(barcode, pbow, wpbow, spbow):
#     if(np.size(barcode) > 0):
#         pbow_diagrams  = pbow.transform(barcode)
#         wpbow_diagrams = wpbow.transform(barcode)
#         spbow_diagrams = spbow.transform(barcode)
#         return pbow_diagrams,wpbow_diagrams,spbow_diagrams
#     else:
#         return [],[],[]

# 8. Function to compute Persistence Silhouette
def GetPersSilhouetteFeature(barcode):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        persSilhouette = representations.vector_methods.Silhouette()
        feature_vector = persSilhouette.fit_transform([barcode])[0]
    
    return feature_vector

# 9. Function to compute Topological Vector
def GetTopologicalVectorFeature(barcode):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        topologicalVector = representations.vector_methods.TopologicalVector()
        feature_vector = topologicalVector.fit_transform([barcode])[0]
    
    return feature_vector

# 10. Function to compute Atol
def GetAtolFeature(barcode, qt):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        atol = representations.vector_methods.Atol(quantiser=qt)
        feature_vector = atol.fit_transform([barcode])[0]
    
    return feature_vector

# 11. Function to compute Complex Polynomial
def GetComplexPolynomialFeature(barcode):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        complexPolynomial = representations.vector_methods.ComplexPolynomial()
        feature_vector = complexPolynomial.fit_transform([barcode])[0]
    
    return feature_vector

# 12. Function to compute lifespan curve
def GetPersLifespanFeature(barcode):
    feature_vector = []
    
    if(np.size(barcode) > 0):
        lfsp = pc.Lifespan()
        feature_vector = lfsp.fit_transform(barcode).flatten()
        
    return feature_vector

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
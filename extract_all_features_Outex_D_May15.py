"""
@author: Dashti
"""

import numpy as np
from sklearn.cluster import KMeans
import vectorization as ex
from multiprocessing import Pool
import glob
import os
from skimage import io
import pandas as pd
import gudhi as gd


def concat_norm_and_opp_export_csv(output_path, feat_name, train_d0, train_d1, test_d0, test_d1, train_opp_d0, train_opp_d1, test_opp_d0, test_opp_d1):
    train_dgm0 = [np.concatenate([dgm0,dgm1]) for dgm0,dgm1 in zip(train_d0, train_opp_d0)]
    train_dgm1 = [np.concatenate([dgm0,dgm1]) for dgm0,dgm1 in zip(train_d1, train_opp_d1)]
    test_dgm0 = [np.concatenate([dgm0,dgm1]) for dgm0,dgm1 in zip(test_d0, test_opp_d0)]
    test_dgm1 = [np.concatenate([dgm0,dgm1]) for dgm0,dgm1 in zip(test_d1, test_opp_d1)]
    
    np.savetxt(f'{output_path}\\{feat_name}_train_feat_pd0.csv', train_dgm0, delimiter=",")
    np.savetxt(f'{output_path}\\{feat_name}_train_feat_pd1.csv', train_dgm1, delimiter=",")
    np.savetxt(f'{output_path}\\{feat_name}_test_feat_pd0.csv', test_dgm0, delimiter=",")
    np.savetxt(f'{output_path}\\{feat_name}_test_feat_pd1.csv', test_dgm1, delimiter=",")

def e_index(x):
    if ((len(str(x))>4)and(str(x)[-4] =='e')):
        y = int(str(x)[-3:])  
    else:
        y = len(str(x))
    return y

def to_float32(dgm):
    y = max([max([e_index(c) for c in cp]) for cp in dgm] + [max([e_index(c) for c in cp]) for cp in dgm])
    n = max(0, y-38+1)
    
    return [[np.float32(x/10**n) for x in cp] for cp in dgm]


if __name__ == '__main__':
    
    # For Outex dataset
    
    folder = 'Datasets/Outex-TC-00024/images'
    images_names = os.listdir(folder)
    images_names = list(filter(lambda x : x[0]!='.', images_names))
    
    images_matrixes = np.array(list(map(lambda x : io.imread(folder+'/'+x), images_names)), dtype=float)
    
    train_names = pd.read_csv("Datasets/Outex-TC-00024/000/train.txt", sep=" ", usecols=[0]).to_numpy().flatten().tolist()
    train_labels = pd.read_csv("Datasets/Outex-TC-00024/000/train.txt", sep=" ", usecols=[1]).to_numpy().flatten().tolist()
    test_names = pd.read_csv("Datasets/Outex-TC-00024/000/test.txt", sep=" ", usecols=[0]).to_numpy().flatten().tolist()
    test_labels = pd.read_csv("Datasets/Outex-TC-00024/000/test.txt", sep=" ", usecols=[1]).to_numpy().flatten().tolist()
    
    train_indexes = list(map(lambda x : images_names.index(x), train_names))
    test_indexes = list(map(lambda x : images_names.index(x), test_names))

    images_gudhi = np.array(list(map(lambda x : x.reshape(128*128,1), images_matrixes)))
    train_gudhi =  images_gudhi[train_indexes]
    test_gudhi = images_gudhi[test_indexes]
    
    train_gudhi_opp =  255-train_gudhi
    test_gudhi_opp = 255-train_gudhi
    
    cub_filtration = lambda image : gd.CubicalComplex(dimensions = [128,128], top_dimensional_cells=image)
    calculate_pd = lambda image : cub_filtration(image).persistence()
    
    train_pds = list(map(calculate_pd, train_gudhi))
    test_pds = list(map(calculate_pd, test_gudhi))
    train_pds_opp = list(map(calculate_pd, train_gudhi_opp))
    test_pds_opp = list(map(calculate_pd, test_gudhi_opp))
    
    #The representation module does not deal well with infinity, so we change it by 256.
    infty_proj = lambda x : 256 if ~np.isfinite(x) else x
    
    intervals_of_dim_0 = lambda pd : np.array([[x[1][0], infty_proj(x[1][1])]  for x in pd if x[0]==0])
    train_pds_0 = list(map(intervals_of_dim_0, train_pds))
    test_pds_0 = list(map(intervals_of_dim_0, test_pds))
    train_pds_opp_0 = list(map(intervals_of_dim_0, train_pds_opp))
    test_pds_opp_0 = list(map(intervals_of_dim_0, test_pds_opp))
    
    intervals_of_dim_1 = lambda pd : np.array([[x[1][0], infty_proj(x[1][1])]  for x in pd if x[0]==1])
    train_pds_1 = list(map(intervals_of_dim_1, train_pds))
    test_pds_1 = list(map(intervals_of_dim_1, test_pds))
    train_pds_opp_1 = list(map(intervals_of_dim_1, train_pds_opp))
    test_pds_opp_1 = list(map(intervals_of_dim_1, test_pds_opp))
    
    output_path = 'E:\\Reseach\\Survey of Persistent Barcode Vectorization\\Exported_Features\\Outex-TC-00024'
    
    np.savetxt(f'{output_path}\\train_labels.csv', train_labels, delimiter=",", fmt='%s')
    np.savetxt(f'{output_path}\\test_labels.csv', test_labels, delimiter=",", fmt='%s')
    
    # Extract Betti Curve features
    res = 100
    train_Btt_0 = [ex.GetBettiCurveFeature(pd, res) for pd in train_pds_0]
    train_Btt_1 = [ex.GetBettiCurveFeature(pd, res) for pd in train_pds_1]
    test_Btt_0 = [ex.GetBettiCurveFeature(pd, res) for pd in test_pds_0]
    test_Btt_1 = [ex.GetBettiCurveFeature(pd, res) for pd in test_pds_1]
    train_Btt_opp_0 = [ex.GetBettiCurveFeature(pd, res) for pd in train_pds_opp_0]
    train_Btt_opp_1 = [ex.GetBettiCurveFeature(pd, res) for pd in train_pds_opp_1]
    test_Btt_opp_0 = [ex.GetBettiCurveFeature(pd, res) for pd in test_pds_opp_0]
    test_Btt_opp_1 = [ex.GetBettiCurveFeature(pd, res) for pd in test_pds_opp_1]
    
    concat_norm_and_opp_export_csv(output_path, "bettiCurve", train_Btt_0, train_Btt_1, test_Btt_0, test_Btt_1, train_Btt_opp_0, train_Btt_opp_1, test_Btt_opp_0, test_Btt_opp_1)
    
    
    # Extract PersStats features
    train_Sta_0 = [ex.GetPersStats(pd) for pd in train_pds_0]
    train_Sta_1 = [ex.GetPersStats(pd) for pd in train_pds_1]
    test_Sta_0 = [ex.GetPersStats(pd) for pd in test_pds_0]
    test_Sta_1 = [ex.GetPersStats(pd) for pd in test_pds_1]
    train_Sta_opp_0 = [ex.GetPersStats(pd) for pd in train_pds_opp_0]
    train_Sta_opp_1 = [ex.GetPersStats(pd) for pd in train_pds_opp_1]
    test_Sta_opp_0 = [ex.GetPersStats(pd) for pd in test_pds_opp_0]
    test_Sta_opp_1 = [ex.GetPersStats(pd) for pd in test_pds_opp_1]
    
    concat_norm_and_opp_export_csv(output_path, "persStats", train_Sta_0, train_Sta_1, test_Sta_0, test_Sta_1, train_Sta_opp_0, train_Sta_opp_1, test_Sta_opp_0, test_Sta_opp_1)
    
    # Extract Persistence Images features
    res = [6,6]
    
    train_PI_0 = [ex.GetPersImageFeature(pd,res) for pd in train_pds_0]
    train_PI_1 = [ex.GetPersImageFeature(pd,res) for pd in train_pds_1]
    test_PI_0 = [ex.GetPersImageFeature(pd,res) for pd in test_pds_0]
    test_PI_1 = [ex.GetPersImageFeature(pd,res) for pd in test_pds_1]
    train_PI_opp_0 = [ex.GetPersImageFeature(pd,res) for pd in train_pds_opp_0]
    train_PI_opp_1 = [ex.GetPersImageFeature(pd,res) for pd in train_pds_opp_1]
    test_PI_opp_0 = [ex.GetPersImageFeature(pd,res) for pd in test_pds_opp_0]
    test_PI_opp_1 = [ex.GetPersImageFeature(pd,res) for pd in test_pds_opp_1]
    
    concat_norm_and_opp_export_csv(output_path, "persImage", train_PI_0, train_PI_1, test_PI_0, test_PI_1, train_PI_opp_0, train_PI_opp_1, test_PI_opp_0, test_PI_opp_1)

    # Extract Persistence Entropy features
    train_E_0 = [ex.GetPersEntropyFeature(pd) for pd in train_pds_0]
    train_E_1 = [ex.GetPersEntropyFeature(pd) for pd in train_pds_1]
    test_E_0 = [ex.GetPersEntropyFeature(pd) for pd in test_pds_0]
    test_E_1 = [ex.GetPersEntropyFeature(pd) for pd in test_pds_1]
    train_E_opp_0 = [ex.GetPersEntropyFeature(pd) for pd in train_pds_opp_0]
    train_E_opp_1 = [ex.GetPersEntropyFeature(pd) for pd in train_pds_opp_1]
    test_E_opp_0 = [ex.GetPersEntropyFeature(pd) for pd in test_pds_opp_0]
    test_E_opp_1 = [ex.GetPersEntropyFeature(pd) for pd in test_pds_opp_1]
    
    concat_norm_and_opp_export_csv(output_path, "persEntropy", train_E_0, train_E_1, test_E_0, test_E_1, train_E_opp_0, train_E_opp_1, test_E_opp_0, test_E_opp_1)

    # Extract Life Span curve features
    train_Life_0 = [ex.GetPersLifespanFeature(pd) for pd in train_pds_0]
    train_Life_1 = [ex.GetPersLifespanFeature(pd) for pd in train_pds_1]
    test_Life_0 = [ex.GetPersLifespanFeature(pd) for pd in test_pds_0]
    test_Life_1 = [ex.GetPersLifespanFeature(pd) for pd in test_pds_1]
    train_Life_opp_0 = [ex.GetPersLifespanFeature(pd) for pd in train_pds_opp_0]
    train_Life_opp_1 = [ex.GetPersLifespanFeature(pd) for pd in train_pds_opp_1]
    test_Life_opp_0 = [ex.GetPersLifespanFeature(pd) for pd in test_pds_opp_0]
    test_Life_opp_1 = [ex.GetPersLifespanFeature(pd) for pd in test_pds_opp_1]

    concat_norm_and_opp_export_csv(output_path, "lifeSpanCurve", train_Life_0, train_Life_1, test_Life_0, test_Life_1, train_Life_opp_0, train_Life_opp_1, test_Life_opp_0, test_Life_opp_1)

    # Extract Tropical Coordinates features
    train_T_0 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in train_pds_0]
    train_T_1 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in train_pds_1]
    test_T_0 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in test_pds_0]
    test_T_1 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in test_pds_1]
    train_T_opp_0 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in train_pds_opp_0]
    train_T_opp_1 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in train_pds_opp_1]
    test_T_opp_0 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in test_pds_opp_0]
    test_T_opp_1 = [ex.GetPersTropicalCoordinatesFeature(pd) for pd in test_pds_opp_1]

    concat_norm_and_opp_export_csv(output_path, "tropicalCoordinates", train_T_0, train_T_1, test_T_0, test_T_1, train_T_opp_0, train_T_opp_1, test_T_opp_0, test_T_opp_1)

    # Extract Atol features
    quantiser = KMeans(n_clusters=2, random_state=202006)

    train_A_0 = [ex.GetAtolFeature(pd, quantiser) for pd in train_pds_0]
    train_A_1 = [ex.GetAtolFeature(pd, quantiser) for pd in train_pds_1]
    test_A_0 = [ex.GetAtolFeature(pd, quantiser) for pd in test_pds_0]
    test_A_1 = [ex.GetAtolFeature(pd, quantiser) for pd in test_pds_1]
    train_A_opp_0 = [ex.GetAtolFeature(pd, quantiser) for pd in train_pds_opp_0]
    train_A_opp_1 = [ex.GetAtolFeature(pd, quantiser) for pd in train_pds_opp_1]
    test_A_opp_0 = [ex.GetAtolFeature(pd, quantiser) for pd in test_pds_opp_0]
    test_A_opp_1 = [ex.GetAtolFeature(pd, quantiser) for pd in test_pds_opp_1]
        
    concat_norm_and_opp_export_csv(output_path, "Atol", train_A_0, train_A_1, test_A_0, test_A_1, train_A_opp_0, train_A_opp_1, test_A_opp_0, test_A_opp_1)

    # Extract Persistence Landscapes features
    train_Land_0 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in train_pds_0]
    train_Land_1 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in train_pds_1]
    test_Land_0 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in test_pds_0]
    test_Land_1 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in test_pds_1]
    train_Land_opp_0 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in train_pds_opp_0]
    train_Land_opp_1 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in train_pds_opp_1]
    test_Land_opp_0 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in test_pds_opp_0]
    test_Land_opp_1 = [ex.GetPersLandscapeFeature(pd, num=20) for pd in test_pds_opp_1]

    concat_norm_and_opp_export_csv(output_path, "PersLandscape", train_Land_0, train_Land_1, test_Land_0, test_Land_1, train_Land_opp_0, train_Land_opp_1, test_Land_opp_0, test_Land_opp_1)

    # Extract Persistence Silhouettes features
    train_Sil_0 = [ex.GetPersSilhouetteFeature(pd) for pd in train_pds_0]
    train_Sil_1 = [ex.GetPersSilhouetteFeature(pd) for pd in train_pds_1]
    test_Sil_0 = [ex.GetPersSilhouetteFeature(pd) for pd in test_pds_0]
    test_Sil_1 = [ex.GetPersSilhouetteFeature(pd) for pd in test_pds_1]
    train_Sil_opp_0 = [ex.GetPersSilhouetteFeature(pd) for pd in train_pds_opp_0]
    train_Sil_opp_1 = [ex.GetPersSilhouetteFeature(pd) for pd in train_pds_opp_1]
    test_Sil_opp_0 = [ex.GetPersSilhouetteFeature(pd) for pd in test_pds_opp_0]
    test_Sil_opp_1 = [ex.GetPersSilhouetteFeature(pd) for pd in test_pds_opp_1]

    concat_norm_and_opp_export_csv(output_path, "persSilhouettes", train_Sil_0, train_Sil_1, test_Sil_0, test_Sil_1, train_Sil_opp_0, train_Sil_opp_1, test_Sil_opp_0, test_Sil_opp_1)

    # Extract Carlsson Coordinates features
    train_CC_0 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in train_pds_0]
    train_CC_1 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in train_pds_1]
    test_CC_0 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in test_pds_0]
    test_CC_1 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in test_pds_1]
    train_CC_opp_0 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in train_pds_opp_0]
    train_CC_opp_1 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in train_pds_opp_1]
    test_CC_opp_0 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in test_pds_opp_0]
    test_CC_opp_1 = [ex.GetCarlssonCoordinatesFeature(pd) for pd in test_pds_opp_1]

    concat_norm_and_opp_export_csv(output_path, "carlssonCoordinates", train_CC_0, train_CC_1, test_CC_0, test_CC_1, train_CC_opp_0, train_CC_opp_1, test_CC_opp_0, test_CC_opp_1)

    # Extract Topological Vectors features
    train_TV_0 = [ex.GetTopologicalVectorFeature(pd) for pd in train_pds_0]
    train_TV_1 = [ex.GetTopologicalVectorFeature(pd) for pd in train_pds_1]
    test_TV_0 = [ex.GetTopologicalVectorFeature(pd) for pd in test_pds_0]
    test_TV_1 = [ex.GetTopologicalVectorFeature(pd) for pd in test_pds_1]
    train_TV_opp_0 = [ex.GetTopologicalVectorFeature(pd) for pd in train_pds_opp_0]
    train_TV_opp_1 = [ex.GetTopologicalVectorFeature(pd) for pd in train_pds_opp_1]
    test_TV_opp_0 = [ex.GetTopologicalVectorFeature(pd) for pd in test_pds_opp_0]
    test_TV_opp_1 = [ex.GetTopologicalVectorFeature(pd) for pd in test_pds_opp_1]
    
    concat_norm_and_opp_export_csv(output_path, "topologicalVectors", train_TV_0, train_TV_1, test_TV_0, test_TV_1, train_TV_opp_0, train_TV_opp_1, test_TV_opp_0, test_TV_opp_1)

    # Extract Complex Polynomials (type = R) features
    train_CPR_0 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in train_pds_0])
    train_CPR_1 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in train_pds_1])
    test_CPR_0 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in test_pds_0])
    test_CPR_1 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in test_pds_1])
    train_CPR_opp_0 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in train_pds_opp_0])
    train_CPR_opp_1 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in train_pds_opp_1])
    test_CPR_opp_0 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in test_pds_opp_0])
    test_CPR_opp_1 = to_float32([ex.GetComplexPolynomialFeature(pd) for pd in test_pds_opp_1])
    
    concat_norm_and_opp_export_csv(output_path, "complexPolynomialsTypeR", train_CPR_0, train_CPR_1, test_CPR_0, test_CPR_1, train_CPR_opp_0, train_CPR_opp_1, test_CPR_opp_0, test_CPR_opp_1)

    # Extract Complex Polynomials (type = S) features
    train_CPS_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in train_pds_0])
    train_CPS_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in train_pds_1])
    test_CPS_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in test_pds_0])
    test_CPS_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in test_pds_1])
    train_CPS_opp_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in train_pds_opp_0])
    train_CPS_opp_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in train_pds_opp_1])
    test_CPS_opp_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in test_pds_opp_0])
    test_CPS_opp_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='S') for pd in test_pds_opp_1])

    concat_norm_and_opp_export_csv(output_path, "complexPolynomialsTypeS", train_CPS_0, train_CPS_1, test_CPS_0, test_CPS_1, train_CPS_opp_0, train_CPS_opp_1, test_CPS_opp_0, test_CPS_opp_1)

    # Extract Complex Polynomials (type = T) features
    train_CPT_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in train_pds_0])
    train_CPT_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in train_pds_1])
    test_CPT_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in test_pds_0])
    test_CPT_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in test_pds_1])
    train_CPT_opp_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in train_pds_opp_0])
    train_CPT_opp_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in train_pds_opp_1])
    test_CPT_opp_0 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in test_pds_opp_0])
    test_CPT_opp_1 = to_float32([ex.GetComplexPolynomialFeature(pd, pol_type='T') for pd in test_pds_opp_1])

    concat_norm_and_opp_export_csv(output_path, "complexPolynomialsTypeT", train_CPT_0, train_CPT_1, test_CPT_0, test_CPT_1, train_CPT_opp_0, train_CPT_opp_1, test_CPT_opp_0, test_CPT_opp_1)


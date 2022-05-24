"""
@author: Dashti
"""

import numpy as np
from sklearn.cluster import KMeans
import vectorization as ex
from multiprocessing import Pool
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def row2matrix(image):
    """
    take a 1x3072 or 1x1024 numpy array representing a color/grey image and reshape it to the
    32x32 format used by scikit-image.
    """
    image_sk = np.zeros([32,32,3], dtype=int)
    for i in range(len(image)):
        x = (i//32)%32
        y = i%32
        z = i//1024
        image_sk[x,y,z]=image[i]
    return image_sk


def color2grey(image):
    """
    take a 1x3072 numpy array representing a color image and gives a 1x1024
    numpy array of integers representing a grey scale image.
    """
    r = image[0:1024]
    g = image[1024:2048]
    b = image[2048:3072]
    #Weighted average of the intensities which produce a nice-viewing gray 
    #scale image.
    grey = 0.2125*r + 0.7154*g + 0.0721*b
    grey = np.array(list(map(lambda x : int(x), grey)))
    return grey


#The row order from the database CIFAR-10 is the transpose respect to the
#one of gudhi. The following functions transform one into the other.
def row_transpose(image):
    """
    Take a 1x1024 int row and change its order to represent the transpose 
    image.
    """
    transpose = np.zeros([1024], dtype=int)
    for i in range(1024):
        x = i//32
        y = i%32
        transpose[y*32+x]=image[i]
    return transpose


def export_feature_parallel_map(func, feat_name, train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool):
    feat_train_pd0 = pool.map(func, train_pd0)
    np.savetxt(f'{output_path}\\{feat_name}Feat_train_pd0.csv', feat_train_pd0, delimiter=",")

    feat_train_pd1 = pool.map(func, train_pd1)
    np.savetxt(f'{output_path}\\{feat_name}Feat_train_pd1.csv', feat_train_pd1, delimiter=",")

    feat_test_pd0 = pool.map(func, test_pd0)
    np.savetxt(f'{output_path}\\{feat_name}Feat_test_pd0.csv', feat_test_pd0, delimiter=",")
    
    feat_test_pd1 = pool.map(func, test_pd1)
    np.savetxt(f'{output_path}\\{feat_name}Feat_test_pd1.csv', feat_test_pd1, delimiter=",")

# Use this function if you need to pass all pds as one input to the input method
# Return one feature vector for all pds
def export_feature_parallel_map2(func, feat_name, train_pds, test_pds, output_path, pool):
    feat_train_pds = pool.map(func, train_pds)
    np.savetxt(f'{output_path}\\{feat_name}Feat_train_pds.csv', feat_train_pds, delimiter=",")

    feat_test_pds = pool.map(func, test_pds)
    np.savetxt(f'{output_path}\\{feat_name}Feat_test_pds.csv', feat_test_pds, delimiter=",")
    

def export_feature_parallel_starmapp(func, arg, feat_name, train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool):
    feat_train_pd0 = pool.starmap(func, [(pd, arg) for pd in train_pd0])
    np.savetxt(f'{output_path}\\{feat_name}Feat_train_pd0.csv', feat_train_pd0, delimiter=",")

    feat_train_pd1 = pool.starmap(func, [(pd, arg) for pd in train_pd1])
    np.savetxt(f'{output_path}\\{feat_name}Feat_train_pd1.csv', feat_train_pd1, delimiter=",")

    feat_test_pd0 = pool.starmap(func, [(pd, arg) for pd in test_pd0])
    np.savetxt(f'{output_path}\\{feat_name}Feat_test_pd0.csv', feat_test_pd0, delimiter=",")
    
    feat_test_pd1 = pool.starmap(func, [(pd, arg) for pd in test_pd1])
    np.savetxt(f'{output_path}\\{feat_name}Feat_test_pd1.csv', feat_test_pd1, delimiter=",")


if __name__ == '__main__':
    
    training_1_path = 'cifar-10-py-original/data_batch_1'
    training_1_dict = unpickle(training_1_path)
    
    training_2_path = 'cifar-10-py-original/data_batch_2'
    training_2_dict = unpickle(training_2_path)
    
    training_3_path = 'cifar-10-py-original/data_batch_3'
    training_3_dict = unpickle(training_3_path)
    
    training_4_path = 'cifar-10-py-original/data_batch_4'
    training_4_dict = unpickle(training_4_path)
    
    training_5_path = 'cifar-10-py-original/data_batch_5'
    training_5_dict = unpickle(training_5_path)
    
    test_path = 'cifar-10-py-original/test_batch'
    test_dict= unpickle(test_path)
    
    
    training_1_list = training_1_dict[b'data']
    training_2_list = training_2_dict[b'data']
    training_3_list = training_3_dict[b'data']
    training_4_list = training_4_dict[b'data']
    training_5_list = training_5_dict[b'data']
    training_list = np.vstack([training_1_list, training_2_list, training_3_list,
                               training_4_list, training_5_list])
    
    training_1_labels = training_1_dict[b'labels']
    training_2_labels = training_2_dict[b'labels']
    training_3_labels = training_3_dict[b'labels']
    training_4_labels = training_4_dict[b'labels']
    training_5_labels = training_5_dict[b'labels']
    training_labels = (training_1_labels + training_2_labels + training_3_labels +
                       training_4_labels + training_5_labels)
    
    test_list = test_dict[b'data']
    test_labels = test_dict[b'labels']
    
    training_gudhi = map(color2grey, training_list)
    training_gudhi = map(row_transpose, training_gudhi)
    training_gudhi = np.array(list(training_gudhi))
    
    training_gudhi = map(color2grey, training_list)
    training_gudhi = map(row_transpose, training_gudhi)
    training_gudhi = np.array(list(training_gudhi))
    
    test_gudhi = map(color2grey, test_list)
    test_gudhi = map(row_transpose, test_gudhi)
    test_gudhi = np.array(list(test_gudhi))
    
    GetCubicalComplexPDs = lambda image : ex.GetCubicalComplexPDs(image, [32,32])
    
    training_pds = list(map(GetCubicalComplexPDs, training_gudhi))
    test_pds = list(map(GetCubicalComplexPDs, test_gudhi))
    
    pd0 = lambda pds : pds[0]
    pd1 = lambda pds : pds[1]
    
    train_pd0 = list(map(pd0, training_pds))
    train_pd1 = list(map(pd1, training_pds))
    
    test_pd0 = list(map(pd0, test_pds))
    test_pd1 = list(map(pd1, test_pds))
    
    output_path = 'E:\\Reseach\\Survey of Persistent Barcode Vectorization\\Exported_Features\\Cifar10'
    np.savetxt(f'{output_path}\\train_labels.csv', training_labels, delimiter=",")
    np.savetxt(f'{output_path}\\test_labels.csv', test_labels, delimiter=",")
    
    pool = Pool(10)
    
    # persStatsFeat
    export_feature_parallel_map(ex.GetPersStats, 'persStats', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)
    
    # persImageFeat
    resolution = [30, 30]
    export_feature_parallel_starmapp(ex.GetPersImageFeature, resolution, 'persImage', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)

    # persLandscapeFeat
    resolution = 100
    export_feature_parallel_starmapp(ex.GetPersLandscapeFeature, resolution, 'persLandscape', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)
    
    # persEntropyFeat
    export_feature_parallel_map2(ex.GetPersEntropyFeature, 'persEntropy', training_pds, test_pds, output_path, pool)
    
    # bettiCurveFeat
    resolution = 100
    export_feature_parallel_starmapp(ex.GetBettiCurveFeature, resolution, 'bettiCurve', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)

    # persSilhouetteFeat
    export_feature_parallel_map(ex.GetPersSilhouetteFeature, 'persSilhouette', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)
    
    # topologicalVectorFeat
    export_feature_parallel_map(ex.GetTopologicalVectorFeature, 'topologicalVector', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)
    
    # atolFeat
    quantiser = KMeans(n_clusters=2, random_state=202006)
    export_feature_parallel_starmapp(ex.GetAtolFeature, quantiser, 'atol', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)
    
    # complexPolynomialFeat
    export_feature_parallel_map(ex.GetComplexPolynomialFeature, 'complexPolynomial', train_pd0, train_pd1, test_pd0, test_pd1, output_path, pool)
    
    # carlssonCoordinatesFeat
    export_feature_parallel_map2(ex.GetCarlssonCoordinatesFeature, 'carlssonCoordinates', training_pds, test_pds, output_path, pool)
    
    # persLifespanFeat
    export_feature_parallel_map2(ex.GetPersLifespanFeature, 'persLifespan', training_pds, test_pds, output_path, pool)

    pool.close()

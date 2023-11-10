import os
import pickle
import numpy as np
from fashion_mnist import mnist_reader
import vectorization as vect
import os
from skimage import io
import pandas as pd

#Generate and save persistence diagrams

folder = 'Outex-TC-00024/data/images'
path_diag = "Outex-TC-00024/pdiagrams/"

images_names = os.listdir(folder)
images_names = list(filter(lambda x : x[0]!='.', images_names))

images_matrixes = np.array(list(map(lambda x : io.imread(folder+'/'+x), images_names)), dtype=float)

path = "Outex-TC-00024/data/000/"
train_names = pd.read_csv(path + "train.txt", sep=" ", usecols=[0]).to_numpy().flatten().tolist()
train_labels = pd.read_csv(path + "train.txt", sep=" ", usecols=[1]).to_numpy().flatten().tolist()
test_names = pd.read_csv(path + "test.txt", sep=" ", usecols=[0]).to_numpy().flatten().tolist()
test_labels = pd.read_csv(path + "test.txt", sep=" ", usecols=[1]).to_numpy().flatten().tolist()

train_indexes = list(map(lambda x : images_names.index(x), train_names))
test_indexes = list(map(lambda x : images_names.index(x), test_names))

images_gudhi = np.array(list(map(lambda x : x.reshape(128*128,1), images_matrixes)))
gudhi_train =  images_gudhi[train_indexes]
gudhi_test = images_gudhi[test_indexes]

gudhi_train_opp =  255 - gudhi_train
gudhi_test_opp = 255 - gudhi_test

for i in range(1360):
    dgms = vect.GetCubicalComplexPDs(img=gudhi_train[i], img_dim=[128,128])
    np.savetxt(path_diag + "l_d0_"+str(i),dgms[0])
    np.savetxt(path_diag + "l_d1_"+str(i),dgms[1])
    
    dgms = vect.GetCubicalComplexPDs(img=gudhi_train_opp[i], img_dim=[128,128])
    np.savetxt(path_diag + "u_d0_"+str(i),dgms[0])
    np.savetxt(path_diag + "u_d1_"+str(i),dgms[1])
    
for i in range(1360):
    j = 1360+i
    dgms = vect.GetCubicalComplexPDs(img=gudhi_test[i], img_dim=[128,128])
    np.savetxt(path_diag + "l_d0_"+str(j),dgms[0])
    np.savetxt(path_diag + "l_d1_"+str(j),dgms[1])
    
    dgms = vect.GetCubicalComplexPDs(img=gudhi_test_opp[i], img_dim=[128,128])
    np.savetxt(path_diag + "u_d0_"+str(j),dgms[0])
    np.savetxt(path_diag + "u_d1_"+str(j),dgms[1])
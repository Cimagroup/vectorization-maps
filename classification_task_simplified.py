import numpy as np
import gudhi as gd
import gudhi.representations


#%%
#Loading the data as proposed in the readme
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

#%%
#Select images with the "horse" (7) and the "ship" (8) label 
indexes = list(filter(lambda i : (training_labels[i]==7)|(training_labels[i]==8), range(len(training_labels))))
training_labels = [training_labels[i] for i in indexes]
training_list = training_list[indexes,:]

indexes = list(filter(lambda i : (test_labels[i]==7)|(test_labels[i]==8), range(len(test_labels))))
test_labels = [test_labels[i] for i in indexes]
test_list = test_list[indexes,:]

#%%
#Transforming each color image into a grey scale image with the gudhi format
def color2grey(image):
    """
    take a 1x3072 numpy array representing a color image and gives a 1x1024
    numpy array of integers representing a gray scale image.
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

training_gudhi = map(color2grey, training_list)
training_gudhi = map(row_transpose, training_gudhi)
training_gudhi = np.array(list(training_gudhi))
test_gudhi= map(color2grey, test_list)
test_gudhi = np.array(list(map(row_transpose, test_gudhi)))

#%%       
#Generate a PD and a landscape from each image

cub_filtration = lambda image : gd.CubicalComplex(dimensions = [32,32], top_dimensional_cells=image)
calculate_dg = lambda image : cub_filtration(image).persistence()

training_pds = list(map(calculate_dg, training_gudhi))
test_pds = list(map(calculate_dg, test_gudhi))

intervals_of_dim_1 = lambda pd : np.array([[x[1][0], x[1][1]]  for x in pd if x[0]==1])
training_pds_1 = list(map(intervals_of_dim_1, training_pds))
test_pds_1 = list(map(intervals_of_dim_1, test_pds))

LS = gd.representations.Landscape(resolution=1000)
training_LS = LS.fit_transform(training_pds_1)
test_LS = LS.fit_transform(test_pds_1)



#%%
#Training the Random Forest
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier = classifier.fit(training_LS, training_labels)

#Showing the accuracy
print("Train accuracy = " + str(classifier.score(training_LS, training_labels)))
print("Test accuracy  = " + str(classifier.score(test_LS, test_labels)))    


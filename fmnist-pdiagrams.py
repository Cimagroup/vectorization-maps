import pickle
import numpy as np
from skimage.feature import canny
from skimage import filters
from skimage import morphology
from scipy.ndimage import distance_transform_bf
from fashion_mnist import mnist_reader
import vectorization as vect


path_feat = "fashion_mnist/features/"
path_diag = "fashion_mnist/pdiagrams/"



#%%
images_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
images_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

n_train = len(images_train)
n_total = len(images_train) + len(images_test)
indexes_train = range(n_train)
indexes_test = range(n_train, n_total)

images = np.vstack([images_train, images_test])
labels = np.array(y_train.tolist() + y_test.tolist())

#%%

root = lambda ima : int(np.sqrt(ima.shape[0]))
#unroot = lambda ima : int((ima.shape[0])**2)
toSquare = lambda ima : ima.reshape([root(ima), root(ima)])
padding = lambda ima : np.pad(ima, ((2,2), (2,2)), 'constant', constant_values=0)
sq=morphology.rectangle(3, 3, dtype='uint8')
median = lambda ima : filters.median(ima, sq)
binarization = lambda ima : 255*(ima>5)
edger = lambda ima : canny(image=ima, low_threshold=20, high_threshold=100)
inverter = lambda ima : np.max(np.float32(ima))-ima
edge_pipeline = lambda ima : inverter(edger(binarization(median((toSquare(ima))))))

edge_images = np.array(list(map(edge_pipeline, images)))

filt_taxi = lambda ima : distance_transform_bf(ima, metric='taxicab')

taxi_complexes = np.array(list(map(filt_taxi, edge_images)))
taxi_complexes_opp = np.array(list(map(inverter, taxi_complexes)))


#%%

for i in range(n_total):
    print(i)
    img_i = taxi_complexes[i]
    n = img_i.shape[0]
    n_square = n**2
    
    dgms = vect.GetCubicalComplexPDs(img=taxi_complexes_opp[i].reshape(n_square,), img_dim=[n,n])
    #we remove the infinity bar
    dgm = dgms[0][:-1]
    np.savetxt(path_diag+ "taxi_u_"+str(i),dgm)
    
#%%

print("DONE")    
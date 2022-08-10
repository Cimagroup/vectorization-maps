# Table of contents
1. [Folders](#folders)
2. [Vectorisation Methods](#vectorisation-methods)
3. [Feature extraction from data](#feature-extraction-from-data)
4. [Classification](#classification)
5. [Examples](#examples)


To install the package: ``pip install . ``


## Folders

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[cifar-10-py-original](https://github.com/Cimagroup/vectorisation-maps/tree/master/cifar-10-py-original) |folder with the dataset cifar-10 |
|[Outex-TC-00024](https://github.com/Cimagroup/vectorisation-maps/tree/master/Outex-TC-00024)             | folder with the dataset Outex-68 and outputs of the experiment. The persistent diagram and features have been deleted, and can be found in OneDrive |
|[fashion_mnist](https://github.com/Cimagroup/vectorisation-maps/tree/master/fashion_mnist)             | folder with the dataset fashion_mnist and outputs of the experiment. The persistent diagram and features have been deleted, and can be found in OneDrive  |
|[Shrec14/pdiagrams](https://github.com/Cimagroup/vectorisation-maps/tree/master/Shrec14/pdiagrams) | folder with the dataset shrec14 and outputs of the experiment. The persistent diagram and features have been deleted, and can be found in OneDrive |
| [WebApp](https://github.com/Cimagroup/vectorisation-maps/tree/master/WebApp) | implementation of the web app of this project |

## Vectorisation Methods

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[Vectorisation/Get(method)](https://github.com/Cimagroup/vectorisation-maps/tree/master/vectorization) | implementation of the method in python |
|[Vectorisation/float64to32](https://github.com/Cimagroup/vectorisation-maps/blob/master/vectorization/float64to32.py) | normalisation used for the complex polynomials |
|[persistence_curves.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/persistence_curves.py)| script defyining betti curve, entropy and lifespan curve|
|[TEST_extract_featurized_barcodes.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/TEST_extract_featurized_barcodes.ipynb)| test to check the features and the cubical complex fucntion works well. It uses the image [TEST_image.pgm](https://github.com/Cimagroup/vectorisation-maps/blob/master/TEST_image.pgm)|


## Feature extraction from data

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[PARALLEL_extract_all_features.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/extract_all_features_parallel.py) | old script for extracting features from the cifar10 database (in paralell) |
|[extract_featurized_barcodes.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/extract_featurized_barcodes.py) | script with functions for all vectorization methods |
|[Test_extract_featurized_barcodes.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/Test_extract_featurized_barcodes.ipynb)| test for functions defined in extract_featurized_barcodes.py|
|[OUTEX_pdiagrams.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX_pdiagrams.py) | script to obtain the persistent diagrams from the OUTEX database |
|[OUTEX_feature_extraction.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX_feature_extraction.py) | script to obtain all the features  from the OUTEX database |
|[fashionMNIST_pdiagrams.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_pdiagrams.ipynb) | notebook to obtain the persistent diagrams and features from the fashion_mnist database |
|[SHREC14_features_computation.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/SHREC14_features_computation.ipynb) | notebook to obtain 3D points from Shrec14 dataset, apply VR-filtration, and compute features |

## Classification
| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[OUTEX68_classification.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX68_classification.py) | Function to classify OUTEX68. Both, the feature and the machine learning method can be given as an input.|
|[OUTEX68_parameter_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX68_parameter_optimisation.py) | Function to find the best parameters for each method when classifying OUTEX68.|
|[OUTEX10_classification.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX10_classification.py) | Function to classify OUTEX10. Both, the feature and the machine learning method can be given as an input.|
|[OUTEX10_parameter_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX10_parameter_optimisation.py) | Function to find the best parameters for each method when classifying OUTEX10.|
|[fashionMNIST_classification.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_classification.ipynb) | notebook to classify the fashion_mnist database using random forest |
|[direct_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/direct_optimisation.py) | estimator with a general classifier from already calculated features|
|[OUTEX_tropical_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX_tropical_optimisation.py) | estimator with a specific classifier for Tropical Coordinates and the OUTEX database|

## Examples
| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
| [CIFAR10_example_classificationipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/CIFAR10_example_classification.ipynb)  |   notebook with an example of classification using cifar10                                |
| [OUTEX68_example_classification.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX68_example_classification.ipynb) | notebook with an example of classification using outex-68|
| [fashionMNIST_example_classification.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_example_classification.ipynb)|notebook with an example of classification using fashion-mnist|



# Table of contents
1. [Folders](#folders)
2. [Vectorisation Methods](#vectorisation-methods)
3. [Feature computation from data](#feature-computation-from-data)
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
|[Vectorisation/Get(method)](https://github.com/Cimagroup/vectorisation-maps/tree/master/vectorisation) | implementation of the method in python |
|[Vectorisation/float64to32](https://github.com/Cimagroup/vectorisation-maps/blob/master/vectorisation/float64to32.py) | re-scale of float 64 numbers to transform them into float 32, since sklearn cannot handle float 64. It is needed for Complex Polynomials |
|[Vectorisation/GetNewMethods.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/vectorisation/GetNewMethods.py)| Auxiliary functions for GetPersEntropy, GetLifespanCurve and GetTopologicalVector |
|[Vectorisation/ATS.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/vectorisation/ATS.py)| auxiliary functions for GetTemplateFunctionFeature and GetAdaptativeSystemFeature. The original script can be found [here](https://github.com/lucho8908/adaptive_template_systems) |
|[Vectorisation/bar_cleaner.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/vectorisation/bar_cleaner.py)| it removes the bars with 0 length, which appear in the SHREC14 database |


## Feature computation from data

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[OUTEX_pdiagrams.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX_pdiagrams.py) | script to obtain the persistent diagrams from the OUTEX database |
|[OUTEX_feature_computation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX_feature_computation.py) | script to obtain all the features  from the OUTEX database |
|[SHREC14_features_computation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/SHREC14_features_computation.ipynb) | notebook to obtain 3D points from Shrec14 dataset, apply VR-filtration, and compute features |
|[fashionMNIST_pdiagrams.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_pdiagrams.py) | notebook to obtain the persistent diagrams and features from the fashion_mnist database |
|[fashionMNIST_feature_computation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_feature_computation.py) | script to obtain all the features from the fashionMNIST database |

## Classification
| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[OUTEX68_classification.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX68_classification.py) | Function to classify OUTEX68. Both, the feature and the machine learning method can be given as an input.|
|[OUTEX68_parameter_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX68_parameter_optimisation.py) | Function to find the best parameters for each method when classifying OUTEX68.|
|[OUTEX10_classification.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX10_classification.py) | Function to classify OUTEX10. Both, the feature and the machine learning method can be given as an input.|
|[OUTEX10_parameter_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX10_parameter_optimisation.py) | Function to find the best parameters for each method when classifying OUTEX10.|
|[SHREC14_classification.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/SHREC14_classification.py) | Function to classify SHREC14. Both, the feature and the machine learning method can be given as an input.|
|[SHREC14_parameter_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/SHREC14_parameter_optimisation.py) | Function to find the best parameters for each method when classifying SHREC14.|
|[fashionMNIST_classification.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_classification.py) | Function to classify fasionMNIST. Both, the feature and the machine learning method can be given as an input.|
|[fashionMNIST_parameter_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/fashionMNIST_parameter_optimisation.py) | Function to find the best parameters for each method when classifying fashionMNIST.|
|[direct_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/direct_optimisation.py) | estimator with a general classifier from already calculated features|
|[OUTEX_tropical_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/OUTEX_tropical_optimisation.py) | estimator with a specific classifier for Tropical Coordinates and the OUTEX database|
|[SHREC14_tropical_optimisation.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/SHREC14_tropical_optimisation.py) | estimator with a specific classifier for Tropical Coordinates and the SHREC14 database|


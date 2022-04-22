# Table of contents
1. [Datasets](#datasets)
2. [Feature extraction from data](#feature-extraction-from-data)
3. [Examples](#examples)

## Datasets

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[cifar-10-py-original](https://github.com/Cimagroup/vectorisation-maps/tree/master/cifar-10-py-original) |folder with the dataset cifar-10 |
|[Outex-TC-00024](https://github.com/Cimagroup/vectorisation-maps/tree/master/Outex-TC-00024)             | folder with the dataset Outex-68 | |[fashion_mnist](https://github.com/Cimagroup/vectorisation-maps/tree/master/fashion_mnist)             | folder with the dataset fashion_mnist |
|


## Feature extraction from data

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[extract_all_features_parallel.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/extract_all_features_parallel.py) | script for extracting all features from the cifar10 database (in paralell) |
|[extract_featurized_barcodes.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/extract_featurized_barcodes.py) | script with functions for all vectorization methods |
|[persistence_curves.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/persistence_curves.py)| script defyining betti curve, entropy and lifespan curve|
|[Test_extract_featurized_barcodes.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/Test_extract_featurized_barcodes.ipynb)|test for functions defined in extract_featurized_barcodes.py|

## Examples
| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
| [example_classification_cifar10.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/example_classification_cifar10.ipynb)  |   notebook with an example of classification using cifar10                                |
|  [example_classification_cifar10_simplified.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/example_classification_cifar10_simplified.py)      |   same example, but written in a script|
|[example_classification_outex68.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/example_classification_outex68.ipynb)|notebook with an example of classification using outex-68|
|[example_pcs.ipynb](https://github.com/Cimagroup/vectorisation-maps/blob/master/example_pcs.ipynb)|example showing how to use the persistence curves defined in [persistence_curves.py](https://github.com/Cimagroup/vectorisation-maps/blob/master/persistence_curves.py)|



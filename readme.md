# A Survey of Vectorization Methods in Topological Data Analysis

This code accompanies the paper 

> D. Ali, A. Asaad, M.-J. Jimenez, V. Nanda, E. Paluzo-Hidalgo, and M. Soriano-Trigueros, “A survey of vectorization methods in topological data analysis,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–14, 2023. doi:[10.1109/TPAMI.2023.3308391](https://doi.org/10.1109/TPAMI.2023.3308391).

Link to the web-app repository: [BRAVA](https://github.com/dashtiali/vectorisation-app)

## Library

This experiment is compatible with the following versions of python >= 3.8 & < 3.11.

A library containing all the vectorization methods can be found in the [vectorization](https://github.com/Cimagroup/vectorization-maps/tree/master/vectorization) folder. 
To install it, download the repository and, in a terminal inside the repository folder, use:

1. `pip install -r requirements.txt`
2. `pip install .`

## Experiments

Experiments with three datasets were developed that you can find in the following scripts:

| Dataset                                                                                |
|----------------------------------------------------------------------------------------|
| [Outex](https://github.com/Cimagroup/vectorization-maps/blob/master/outex.py)          |
| [Fashion MNIST](https://github.com/Cimagroup/vectorization-maps/blob/master/fmnist.py) |
| [Shrec14](https://github.com/Cimagroup/vectorization-maps/blob/master/shrec14.py)      |

The barcodes used in the experiments were calculated using the following scripts:

| Dataset                                                                                          |
|--------------------------------------------------------------------------------------------------|
| [Outex](https://github.com/Cimagroup/vectorization-maps/blob/master/outex-pdiagrams.py)          |
| [Fashion MNIST](https://github.com/Cimagroup/vectorization-maps/blob/master/fmnist-pdiagrams.py) |


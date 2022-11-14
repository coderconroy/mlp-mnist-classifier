# First Principles Multilayer Perceptron for MNIST Classifiction

## Overview

A Multilayer Perceptron (MLP) is a fully-connected feedforward artificial neural network which can be used for multi-class classification. This project contains a MLP implemented using only the NumPy library and applied to the MNIST dataset of handwritten digits. A final accuracy of 98.29% was obtained on the test set.

## Approach

The MNIST handwritten digit dataset contains 60 000 training samples and 10 000 testing samples where each sample is a 28x28 pixel image of a single handwritten digit in the range 0 to 9. The trainig set was split between traning and validation in the ratio 5:1 respectively. The MLP itself was implemented in a modular manner to faciliate simple alteration of network parameters to assist in achieving optimal performance. Several different model variations were trained and compared using the performance on the validation set. The best performing model was then trained on all 60 000 training samples and evaluated using the test set.

## Repository Structure

- *trainig.ipynb* - Training and evaluation of various models on MNIST dataset.
- *model.py* - MLP implementation.
- *utils.py* - Miscellaneous output formatting functions.
- *models/* - Contains snapshots of each trained model.

## Description
This project implements a classifier artificial neural network based on the famous back-propagation algorithm. The neural network includes one hidden layer, and can be ran on datasets that have multiple features and one or multiple outputs. The implementation was done in Python 2.7.

## Requirements
This project runs on python 2.7, and needs Numpy and matplotlib libraries to be installed.

## Details about the implementation
The training script applies the forward pass and the back-propagation on the given dataset, then calculates the average error for all outputs and samples in each epoch.

The testing phase starts by running the forward pass using the weights that generated the smaller error, then uses the output to calculate the accuracy of the model.

The accuracy is calculated by turning the biggest value of the classes (outputs) to 1, and the other ones to 0 in case of multiple classes, and uses a threshold value of 0.5 in case of a unique class.

For example: [ 0.8903803   0.1029485   0.05636533] will turn into [1. 0. 0.]
and: [0.634567] will turn into [1.]

The generated array is then compared to the actual/desired output. We then calculate the ratio of how many times the NN was right, over the number of samples we provided as testing dataset.

## Installation
The script accepts exactly 4 arguments: the first one is the path to the training dataset, the second one is the path to the validation dataset, the third one is the number of features (inputs) the dataset has, the fourth one is the number of classes (outputs) the dataset includes, then the last arguments are the controls (HyperParameters) of the model: Number of epochs, Learning rate, Momentum and Number of hidden nodes in the hidden layer.

To run the training, just use the following command:
```
$ ./train_v4.py <path_to_training_dataset> <path_to_testing_dataset> <number_features> <number_classes> <number_epochs> <learning_rate> <momentum> <number_hiddens>
```

The result shows the errors returned for the last epoch during the training, followed by the accuracy of the model over the testing set.

By the end of the script, two graphs are plotted: the first one is for the training and testing error for each epoch, and the second one shows the accuracy of the model for each epoch.

## Author
Manal Jazouli,
Email: <mjazouli@uoguelph.ca>

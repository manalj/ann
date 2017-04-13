#!/usr/bin/python

import numpy as np
import sys
import matplotlib.pyplot as plt

# Hyper parameters
Alpha = 0.05     # Learning rate
Ro = 0.5         # Momentum
n_hiddens = 7    # Number of hidden nodes

dtype = 'float32'

# Load data from file
def load_data(fileName, n_class, n_feature): # 3, 4
    data = open(fileName).readlines()
    data = data[1:len(data)-1] # Open file and skip first line - the header, and the last line
    i=0
    A = np.empty([len(data), n_feature])
    D = np.empty([len(data), n_class])
    for line in data:
        vals = line.split( )
        A[i] = np.array(vals[0:n_feature])  # Taking the features columns
        D[i] = np.array(vals[n_feature:n_class+n_feature]) # Taking the classes columns
        i+=1
    A = (A - A.min(axis=0)) / (A.max(axis=0) - A.min(axis=0))# Normalize data
    return [A,D] # Inputs and Outputs

# Initialize weights and biases with random values in [0,1]
def init_model(n_inputs, n_hiddens, n_outputs):
    np.random.seed(1) # Feeding a seed to the random function to get the same random weights for testing purposes
    # Weights
    V = np.random.rand(n_inputs, n_hiddens)
    V_delta = np.zeros((n_inputs, n_hiddens))
    W = np.random.rand(n_hiddens, n_outputs)
    W_delta = np.zeros((n_hiddens, n_outputs))

    # Biases
    Gamma = np.random.rand(n_outputs)
    Omega = np.random.rand(n_hiddens)

    return [V, V_delta, W, W_delta, Gamma, Omega]

# sigmoid function
def sigmoid(arg):
    return (1 / (1 + np.exp(-arg)))

def sigmoid_deriv(arg):
    return (arg * (1 - arg))

def forward_pass(A, V, W, Gamma, Omega):
    B = sigmoid(np.dot(A, V) + Omega) # Dot product of inputs and V weights plus bias
    C = sigmoid(np.dot(B, W) + Gamma) # Dot product of hiddens and W weights plus bias
    return [B, C]

def back_prop(V, V_delta, W, W_delta, Gamma, Omega, A, B, C, D):

    # Calculate the errors
    E, Eps_pattern = calculate_error(C, D)
    F = sigmoid_deriv(B) * np.dot(E, np.transpose(W))

    # Update the weights #
    # B to C #
    if Ro > 0:
        W += Alpha * np.dot(np.transpose(B), E) + Ro * W_delta
        W_delta = Alpha * np.dot(np.transpose(B), E)
    else:
        W += Alpha * np.dot(np.transpose(B), E)
    Gamma += Alpha * np.mean(E, axis=0)

    # A to B #
    if Ro > 0:
        V += Alpha * np.dot(np.transpose(A), F) + Ro * V_delta
        V_delta = Alpha * np.dot(np.transpose(A), F)
    else:
        V += Alpha * np.dot(np.transpose(A), F)
    Omega += Alpha * np.mean(F, axis=0)

    return [Eps_pattern, V, W, Gamma, Omega]

def calculate_error(C, D):
    E = sigmoid_deriv(C) * (D - C)
    Eps_pattern = np.mean((D-C)**2, axis=0)
    return [E, Eps_pattern]


def train_model(fileName, valFileName, n_features, n_classes, n_epochs, Alpha, Ro, n_hiddens):

    train_eps = np.zeros(n_epochs, dtype=dtype)
    val_eps   = np.zeros(n_epochs, dtype=dtype)
    accuracy    = np.zeros(n_epochs, dtype=dtype)

    A, D = load_data(fileName, n_classes, n_features)
    A_val, D_val = load_data(valFileName, n_classes, n_features)
    V, V_delta, W, W_delta, Gamma, Omega = init_model(n_features, n_hiddens, n_classes) # Initialize the weights according to the given data length
    Eps = float('inf')                                                                  # (n_inputs, n_hiddens, n_outputs)
    epoch = 0
    t = 0.0001          # Minimum error to tolerate

    # START TRAINING
    while (epoch < n_epochs and Eps > t):
        TP, TN, FP, FN = (np.zeros((n_classes), dtype=dtype) for i in range(4))
        # Training set
        B, C = forward_pass(A, V, W, Gamma, Omega)
        # Validation set
        B_val, C_val = forward_pass(A_val, V, W, Gamma, Omega)
        # Calculate error and update weights for training set
        Eps, updated_V, updated_W, updated_Gamma, updated_Omega = back_prop(V, V_delta, W, W_delta, Gamma, Omega, A, B, C, D)
        # Calculate error for validation set
        null, Eps_val = calculate_error(C_val, D_val)
        Eps = np.mean(Eps)
        Eps_val = np.mean(Eps_val)

        # Push errors into arrays for plotting
        train_eps[epoch] = Eps
        val_eps[epoch]   = Eps_val

        if n_classes==1:
            C_val[C_val >= 0.5] = 1 # Turn values to 1 if bigger than 0.5, and to 0 otherwise
            C_val[C_val < 0.5] = 0
        else:
            C_val = (C_val == C_val.max(axis=1)[:,None]).astype(float) # Turn the biggest value to 1 and others to 0

        ## Calculate T/F P/N - statistical indexes
        for output in range(n_classes):
            # predict a label of 1 and the true label is 1
            TP[output] = np.sum(np.logical_and(C_val[:,output] == 1, D_val[:,output] == 1))
            # predict a label of 0 and the true label is 0
            TN[output] = np.sum(np.logical_and(C_val[:,output] == 0, D_val[:,output] == 0))
            # predict a label of 1 and the true label is 0
            FP[output] = np.sum(np.logical_and(C_val[:,output] == 1, D_val[:,output] == 0))
            # predict a label of 0 and the true label is 1
            FN[output] = np.sum(np.logical_and(C_val[:,output] == 0, D_val[:,output] == 1))

        # Measuring performance
        ## Accuracy == sum((TP + TN) / (TP + TN + FP + FN))/n_classes
        accuracy[epoch] = (np.sum(np.true_divide(np.add(TP, TN), np.add(np.add(TP, TN), np.add(FP, FN))))*100)/n_classes

        if epoch == (n_epochs-1):
            print "Epoch N: %d Training Error : %6.4f   Validation Error : %6.4f" % (epoch, Eps, Eps_val)
            print "Accuracy : %.2f %% \n" % accuracy[epoch] # Accuracy is number of (true positive + true negative) / number of samples

        epoch += 1

    # Replace nan values by 0
    accuracy  = np.nan_to_num(accuracy)

    return train_eps, val_eps, accuracy

    ## Plot graph for training_error vs validation_error
    plt.figure(1)
    plt.plot(train_eps, '-r', label="Training data")
    plt.plot(val_eps, '-b', label="Validation data")
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    # Plot graph for accuracy of validation set
    plt.figure(2)
    plt.plot(accuracy, '-g', label="Accuracy of the model on validation data")
    # plt.legend(loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.show()

from collections import OrderedDict

import numpy as np

from layers_solution import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax, softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape,
                 n_output_classes,
                 conv1_channels,
                 conv2_channels,
                 reg=0):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        height, width, channels = input_shape
        self.input_shape = input_shape
        self.n_output_classes = n_output_classes
        self.reg = reg

        self.layers = OrderedDict([
            ('conv1', ConvolutionalLayer(channels, conv1_channels, 3, 1)),
            ('act1', ReLULayer()),
            ('pool1', MaxPoolingLayer(4, 4)),

            ('conv2', ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)),
            ('act2', ReLULayer()),
            ('pool2', MaxPoolingLayer(4, 4)),

            ('flat3', Flattener()),
            # w&h after last layer equal to 2, so we can hardcode 2 * 2 * channels
            ('fc3', FullyConnectedLayer(2 * 2 * conv2_channels, n_output_classes)),
        ])


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        self.zero_grad()

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        f_value = X
        for layer in self.layers.values():
            f_value = layer.forward(f_value)

        loss, dpred = softmax_with_cross_entropy(f_value, y)

        for layer in reversed(self.layers.values()):
            dpred = layer.backward(dpred)

        for layer_name, layer in self.layers.items():
            if layer_name.startswith('fc'):
                for param_name, param in layer.params().items():
                    reg_loss, reg_grad = l2_regularization(param.value, self.reg)
                    loss += reg_loss
                    param.grad += reg_grad

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        f_value = X
        for layer in self.layers.values():
            f_value = layer.forward(f_value)
        pred = softmax(f_value)
        return pred.argmax(axis=1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for layer_name, layer in self.layers.items():
            for param_name, param in layer.params().items():
                result[layer_name + '_' + param_name] = param

        return result

    def zero_grad(self):
        for name, param in self.params().items():
            param.grad = np.zeros_like(param.grad, dtype='float64')

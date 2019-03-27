import numpy as np

from layers_solution import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.l1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.a1 = ReLULayer()

        self.l2 = FullyConnectedLayer(hidden_layer_size, n_output)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.zero_grad()

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        preds = self.l1.forward(X)
        preds = self.a1.forward(preds)
        preds = self.l2.forward(preds)
        loss, dpred = softmax_with_cross_entropy(preds, y)
        
        l2grad = self.l2.backward(dpred)
        a1grad = self.a1.backward(l2grad)
        l1grad = self.l1.backward(a1grad)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = softmax(pred)
        return pred.argmax(axis=1)

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for name, param in self.l1.params().items():
            result['l1' + name] = param

        for name, param in self.l2.params().items():
            result['l2' + name] = param
        return result

    def zero_grad(self):
        for param in self.params().values():
            param.grad = np.zeros_like(param.grad.shape)

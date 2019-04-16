import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    x = predictions.copy()
    if len(predictions.shape) == 1:
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    if len(probs.shape) == 1:
        return -np.log(probs[target_index])
    else:
        n_samples = probs.shape[0]
        return np.mean(-np.log(probs[np.arange(n_samples), target_index]))


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = reg_strength * 2 * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probes = softmax(predictions)
    loss = cross_entropy_loss(probes, target_index)
    dprediction = probes.copy()

    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        n_samples = probes.shape[0]
        dprediction[np.arange(n_samples), target_index] -= 1
        dprediction /= n_samples

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.mask_ = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.mask_ = X > 0
        return X * self.mask_

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = self.mask_ * d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = np.copy(X)
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = d_out.sum(axis=0)[None, :]
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, verbose=False):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.verbose = verbose
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 1 + self.padding * 2
        out_width = width - self.filter_size + 1 + self.padding * 2

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        if self.padding:
            X_padding = np.zeros((batch_size, height + self.padding * 2,
                                  width + self.padding * 2, channels))
            X_padding[:, self.padding:height + self.padding, self.padding:width + self.padding, :] = X
            self.X = X_padding.copy()
        else:
            self.X = X.copy()

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                current_X = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :].reshape((batch_size, -1))
                current_W = self.W.value.reshape((-1, self.out_channels))
                if self.verbose:
                    print('X', current_X.shape)
                    print('W', current_W.shape)
                out[:, y, x, :] = np.dot(current_X, current_W) + self.B.value

        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_input = np.zeros_like(self.X)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                current_X = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :].reshape((batch_size, -1))
                current_W = self.W.value.reshape((-1, self.out_channels))
                if self.verbose:
                    print('X', current_X.shape)
                    print('W', current_W.shape)

                self.W.grad += np.dot(current_X.T, d_out[:, y, x, :]).reshape(self.W.value.shape)
                self.B.grad += d_out[:, y, x, :].sum(axis=0)
                d_input[:, y:y + self.filter_size, x: x + self.filter_size, :] += \
                    np.dot(d_out[:, y, x, :], current_W.T).reshape((batch_size, self.filter_size,
                                                                    self.filter_size, channels))
        if self.padding:
            d_input = d_input[:, self.padding:height-self.padding, self.padding:width-self.padding, :]

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros((batch_size, out_height, out_width, channels))
        self.X = X.copy()

        for y in range(out_height):
            for x in range(out_width):
                current_X = self.X[:,
                                   y * self.stride:y * self.stride + self.pool_size,
                                   x * self.stride:x * self.stride + self.pool_size,
                                   :]
                out[:, y, x, :] = np.max(current_X, axis=(1, 2))
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        d_input = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                current_X = self.X[:,
                                   y * self.stride:y * self.stride + self.pool_size,
                                   x * self.stride:x * self.stride + self.pool_size,
                                   :]
                d_input_part = np.equal(np.ones_like(current_X) *
                                        current_X.max(axis=(1, 2)).reshape(batch_size, 1, 1, channels),
                                        current_X) * d_out[:, y, x, :].reshape(batch_size, 1, 1, channels)
                d_input[:,
                        y * self.stride:y * self.stride + self.pool_size,
                        x * self.stride:x * self.stride + self.pool_size,
                        :] += d_input_part
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}

import numpy as np
import random


class Network:
    def __init__(self, neurons_per_layer):
        self.layers = len(neurons_per_layer)
        self.biases = [np.random.randn(y, 1) for y in neurons_per_layer[1:]]
        weights_rows = neurons_per_layer[1:]
        weights_columns = neurons_per_layer[:-1]
        self.weights = [np.random.randn(rows, column) for rows, column in zip(weights_rows, weights_columns)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
       gradient descent.  The ``training_data`` is a list of tuples
       ``(x, y)`` representing the training inputs and the desired
       outputs.  The other non-optional parameters are
       self-explanatory.  If ``test_data`` is provided then the
       network will be evaluated against the test data after each
       epoch, and partial progress printed out.  This is useful for
       tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.__stochastic_gradient_descent(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def __stochastic_gradient_descent(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        mini_batch_bias_gradients = [np.zeros(b.shape) for b in self.biases]
        mini_batch_weights_gradients = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            bias_gradient_x, weights_gradient_x = self.__back_propagation(x, y)
            mini_batch_bias_gradients = [existing_bias_total + x_train_bias
                                         for existing_bias_total, x_train_bias
                                         in zip(mini_batch_bias_gradients, bias_gradient_x)]
            mini_batch_weights_gradients = [existing_weight_total + x_train_weights
                                            for existing_weight_total, x_train_weights
                                            in zip(mini_batch_weights_gradients, weights_gradient_x)]
        # apply gradient descent result of this mini-batch to the biases and weights
        self.biases = [bias - ((learning_rate / len(mini_batch)) * bias_gradient)
                       for bias, bias_gradient
                       in zip(self.biases, mini_batch_bias_gradients)]
        self.weights = [weights - ((learning_rate / len(mini_batch)) * weights_gradient)
                        for weights, weights_gradient
                        in zip(self.weights, mini_batch_weights_gradients)]

    def __back_propagation(self, x_train, y_train):
        """
            This method is used to calculate the gradient descent for each training input x_train and expected output y_train
            steps:

            1) Feed-forward to calculate the predictions
            2) use back propagation to calculate the gradient descent
                example : calculations for one neuron k from l-1 layer to j neuron of l layer
                    a(l-1)k ------------> alj(blj/bias)
                            wljk(weight)
                    Formulas:
                        Cost_calculations:
                            ** zlj = wljk.a[l-1]k + blj
                            ** alj = sigmoid_derivative(zlj)
                            ** cost = 1/2(alj-y_train)^2
                        Gradient_derivatives
                            ** change in cost for change in bias
                                b_change = sigmoid_derivative(zlj)*cost_derivative(alj,y_train)
                            ** change wrt weight = a[l-1]k * b_change
                            ** change wrt previous neuron = wljk * b_change
                                for change in all weights of the layer can done
                                using (wl)T --> transpose of wl as we are back propagating
                            ** For next layer a[l-2] --> a[l-1] we take change cost with respect
                                change in bias and weights of this layer using change cost by changing a[l-1]
        """
        bias_gradients = [np.zeros(layer_bias.shape) for layer_bias in self.biases]
        weight_gradients = [np.zeros(layer_weight.shape) for layer_weight in self.weights]
        activation = x_train
        # memorize activations of each layer "activation[-1] -> gives prediction"
        activations = [activation]
        # memorize the zs=wl.a[l-1]+bl which can be used while back propagation
        zs = []
        # step 1: feed-forward
        for layer_weight, layer_bias in zip(self.weights, self.biases):
            zl = np.dot(layer_weight, activation) + layer_bias
            zs.append(zl)
            activation = self.__sigmoid(zl)
            activations.append(activation)

        # step 2: back-propagation
        # for last layer
        cost_derivative = self.__cost_derivative(activations[-1], y_train)
        sigmoid_derivative = self.__sigmoid_derivative(zs[-1])
        change_in_bias = cost_derivative * sigmoid_derivative
        # we have weights for this layer 2-dim matrix --> "len(change_in_bias) X len(a[l-1])"
        change_in_weight = np.dot(change_in_bias, activations[-2].transpose())
        bias_gradients[-1] = change_in_bias
        weight_gradients[-1] = change_in_weight

        # Continue back-propagation for previous layers
        # l is taken from 2 which is previous to last layer and continues till
        # as xrange start from 0 to n-1, So we are eliminating the first layer [-0] and [-size]
        # l we take negative of it so that we can traverse from previous last to till first layer
        for l in xrange(2, self.layers):
            # change in cost for change a[l-1] for first time and continues to a[l-2]....
            # weight of next layer is taken transpose because we are back-propagating
            error_this_layer = np.dot(self.weights[-l + 1].transpose(), change_in_bias)
            # this layer changes
            sigmoid_derivative = self.__sigmoid_derivative(zs[-l])
            change_in_bias = error_this_layer * sigmoid_derivative
            # we have weights for this layer 2-dim matrix --> "len(change_in_bias) X len(a[l-1])"
            change_in_weight = np.dot(change_in_bias, activations[-l - 1].transpose())
            bias_gradients[-l] = change_in_bias
            weight_gradients[-l] = change_in_weight
        return bias_gradients, weight_gradients

    def __cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y

    def __sigmoid_derivative(self, z):
        """Derivative of the sigmoid function."""
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))

    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.__sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):

        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def __sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

'''
Prashant Sridhar (54565839)
Project 2

This python module is used to create Neural Networks, and provides functionality for forward propagating, testing, and
training. Training uses back propagation to return. Networks is currently set for logarithmic activation functions, with
linear activated outputs.
'''

import numpy as np
import math
from random import shuffle

# Sigmoid and linear activation functions. Returns respective derivative when field is True
def sigmoid(x, derivative = False):
    if not derivative:
        # overflow error when -x <= -710
        if x > 709: return 1
        elif x < -709: return 0
        else: return 1/ (1 + math.exp(-x))
    else:
        return x * (1.0 - x)

def linear(x, derivative = False):
    if not derivative:
        return x
    else:
        return 1

class Network:
    ''' Network class creates an multi-layer feed forward net with an arbitrary number of inputs nodes, output nodes,
        hidden layers, and nodes per hidden layer. Each node can have either a linear or sigmoidal activation function,
        which can be specified in the self.activation array. The network uses backpropagation to learn, with a tunable
        learning rate. Does not utilize momentum. Currently is trained using online learning.'''

    def __init__(self, shape = []):
        ''' Creates an instance of a network, the input list specifies the number of layers and nodes in each layer, i.e.
            Network([2,3,4,1]) would create a network with 2 input nodes, a hidden layer of 3 nodes, a hidden layer of 4
            nodes, and an output layer of 1 node. Each nodes activation can be specified in the activation table, the
            value in table at the node's position determines its activation function.'''

        self.shape = shape
        self.num_outputs = self.shape[-1]
        self.num_inputs = self.shape[0]
        self.weights_shape = [(self.shape[i-1], self.shape[i]) for i in range(1, len(self.shape))]

        # creates a an array representing a nodes, containing the node's most recent output
        self.layers = [np.zeros(size) for size in shape]

        self.activation = [np.zeros(size) for size in shape[:-1]] # speicifies sigmoidal activation for input & hidden nodes
        self.activation.append(np.ones(shape[-1]))  # specifies linear activation for output nodes

        # creates an triply nested array of weights, first layer, then node, then connection. Random initialization between 0-0.1
        self.weights = [np.random.uniform(0.0, 0.1, size) for size in zip(self.shape[:-1], self.shape[1:])]

    def activate(self, i, j, total, derivative = False):
        ''' Activates a node using its position in the node table, the number to be activated, and whether it wants the
            derivative or true value returned.'''

        if self.activation[i][j] == 0:      # hidden layer using sigmoid
            #print("{}, {} = {}".format(i, j, total))
            return sigmoid(total, derivative)
        if self.activation[i][j] == 1:      # output layer using linear
            return linear(total, derivative)

    def forward_prop(self, input):
        ''' Feeds values forward through the network, storing node outputs as it goes. Input is assumed to be the
            same size as the number of nodes in the input layer. Returns the final output values(s). '''

        #load inputs into input layer
        for i in range(self.num_inputs):
            self.layers[0][i] = input[i]

        for i in range(1, len(self.shape)): # for each layer
            previous_layer = self.layers[i - 1]
            nodes_in, nodes = self.shape[i - 1], self.shape[i]

            for j in range(nodes):          # for each node in the current layer
                total = 0.0                 # no bias
                for k in range(nodes_in):   # for each incoming connection
                    total += previous_layer[k] * self.weights[i - 1][k][j]  # sum each incoming value * weight_connection

                self.layers[i][j] = self.activate(i, j, total)

        return self.layers[-1]

    def back_prop(self, target, learning_rate=0.1):
        ''' Back-propagates through a network. Arguments are the desired output and the learning rate. The networks true
            output is already stored in the output layers nodes. Target is assumed to be the same length as the number of
            output nodes in the network. Returns the error of the error of the network's output.'''

        delta_list = []
        delta = np.zeros(self.num_outputs)
        output_error = 0

        # backprop over the output layer
        for k in range(self.num_outputs):
            output = self.layers[-1][k]
            error = target[k] - output
            output_error += error ** 2
            delta[k] = self.activate(-1, k, output, True) * error

        delta_list.append(delta)

        # backprop over hidden and input layers
        for i in reversed(range(1, len(self.shape) - 1)): # for each layer
            nodes, nodes_out = self.shape[i], self.shape[i + 1]
            next_delta = np.zeros(nodes)

            for j in range(nodes):  # for each node
                error = 0.0
                for k in range(nodes_out):  # for each connection
                    error += delta[k] * self.weights[i][j][k]

                output = self.layers[i][j]
                next_delta[j] = self.activate(i, j, output, True) * error

            delta = next_delta # update delta for next iteration
            delta_list = [delta] + delta_list   # add delta to the complete list
            
        self.update_weights(delta_list, learning_rate)
        return output_error


    def update_weights(self, delta_list, learning_rate):
        ''' Updates the network weights after backpropagation. Takes a doubly nested list containing weight updates for
            each node, and the learning rate.  '''

        for i in range(len(self.weights_shape)): # for every set of weights
            nodes_in, nodes_out = self.weights_shape[i]
            delta = delta_list[i]
            for j in range(nodes_in):               # for for each node
                for k in range(nodes_out):          # for each output weight on in input node
                    change = delta[k] * self.layers[i][j]
                    self.weights[i][j][k] += learning_rate * change


    def train(self, data, iterations = 1):
        ''' Trains network on supplied data. Data input is a list of zipped input and target value lists. '''

        print_frequency = 250

        for i in range(iterations+1):
            shuffle(data)
            sum_error = 0
            for input, target in data:
                output = self.forward_prop(input)
                error = self.back_prop(target, 0.1)
                sum_error += error
                # print(self.layers)
                if i%print_frequency == 0:
                    print('Iteration = {}, Error = {:.5f},  Target = {}, Output = {}'.format(i, error, target, output))
            sum_error = sum_error / len(data)
            if i%print_frequency == 0:
                print("Error at training iteration {} = {}".format(i, sum_error))
            if sum_error < 0.0001: break

    def test(self, data):
        ''' Returns the mean squared error of the network when tested on input data. '''
        sum_error = 0
        for input, target in data:
            output = self.forward_prop(input)
            error = 0
            for i in range(len(output)):
                error += (target[i] - output[i]) ** 2
            sum_error += error
        return sum_error / len(data)

    def output_test(self, data):
        ''' Neatly prints the results of testing the input data on the network, including individual error
            and overall error. '''
        sum_error = 0
        for input, target in data:
            output = self.forward_prop(input)
            error = 0
            for i in range(len(output)):
                error += (target[i] - output[i]) ** 2
            print('Target = {}, Output = {}, Error = {:.5f}'.format(target, output, error))
            sum_error += error
        print("Overall network error is {}".format(sum_error / len(data)))
        #return sum_error / len(data)


def train(shape, iterations, data, learning_rate = 0.1, print_frequency = 250):
    ''' Trains network on supplied data. Data input is a list of zipped input and target value lists. '''

    nn = Network(shape)

    lifetime_error = ['Neural Net {}'.format(shape)]
    for i in range(iterations+1):
        shuffle(data)
        sum_error = 0
        for input, target in data:
            output = nn.forward_prop(input)
            error = nn.back_prop(target, learning_rate)
            sum_error += error
            #if i%print_frequency == 0:
            #    print('Iteration = {}, Error = {:.5f},  Target = {}, Output = {}'.format(i, error, target, output))
        sum_error = sum_error / len(data)
        lifetime_error.append(sum_error)
        if i%print_frequency == 0:
            print("Error at training iteration {} = {}".format(i, sum_error))
        if sum_error < 0.00001: break
    return nn, lifetime_error

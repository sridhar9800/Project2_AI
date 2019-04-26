'''
Prashant Sridhar (54565839)
Project 2

This python module handles the testing of the for neural net training algorithms. Tests data sets using k-fold
cross validation, outputs the networks error throughout training and on the final test set on line graphs, one for
each fold. Output can be saved to a PDF.
'''

import NN
import GA
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import os

def import_data(file_name):
    ''' Imports data points from supplied file, formatting them into data point pairs to be
        used by the algorithms. '''
    fin = open(file_name, 'r')
    input_line = fin.readline()
    data = []
    inputs = 0
    outputs = 0
    read_shape = False

    while input_line:
        input_line = input_line.strip().split(',')
        output_line = fin.readline().strip().split(',')
        if not read_shape:
            inputs = len(input_line)
            outputs = len(output_line)
            read_shape = True
        if input_line == [''] or output_line == ['']: break
        for i in range(len(input_line)): input_line[i] = float(input_line[i])
        for i in range(len(output_line)): output_line[i] = float(output_line[i])
        data.append((input_line, output_line))
        input_line = fin.readline()
    fin.close()
    return data, inputs, outputs, os.path.splitext(file_name)[0][9:] # last return is file name, assumes file is in datasets/ directory


# load data sets from files
data_diabetes = import_data('datasets/diabetes.txt')

# alter these parameters for general testing
dataset, inputs, outputs, data_name = data_diabetes
shuffle(dataset)
shape = [inputs,25,25,outputs]
k = 6   # cannot be larger than 6 without altering the figure's subplot layout

# create folds for cross validation
slices = []
length = int(len(dataset) / k)
for i in range(k-1):
    slices.append(dataset[i*length:(i+1)*length])
slices.append(dataset[(k-1)*length:])

plot_count = 1
fig = plt.figure(figsize=(14, 7))

# for each fold, rotates through testing sets
for i in range(k):
    test_data = slices.pop(0)
    train_data = [j for i in slices for j in i]
    slices.append(test_data)

    # train network, receive error throughout training and the trained network. Uncomment line to train a network
    hero, lifetime_error = NN.train(shape=shape, iterations=200, data=train_data, learning_rate = 0.1, print_frequency=100)
    # hero, lifetime_error = GA.train(shape=shape, mu=20, generations=100, number_possible_parents=10, number_parents=10, quick=True, data=train_data, print_frequency=100)

    test_error = hero.test(test_data)
    print("Testing error on fold {} = {}".format(i+1, test_error))

    # adds the fold's graph to the output figure
    fig.add_subplot(230+plot_count)
    plot_count += 1
    plt.title(lifetime_error[0] + ' on dataset {}'.format(data_name))
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Generation')
    plt.grid(True)
    plt.plot(lifetime_error[1:])
    plt.plot(len(lifetime_error)-1, test_error, 'bs')

plt.tight_layout()
fig.savefig('figures/' + lifetime_error[0] + ' on dataset {}'.format(data_name) + '.pdf')
#plt.show()



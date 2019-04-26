'''
Prashant Sridhar (54565839)
Project 2

This python module is used to train a neural network using the Genetic Algorithm.
'''

import numpy as np
import random
from operator import itemgetter
import NN
import statistics

def create_population(shape, mu):
    ''' Creates the initial population vectors of size mu, based on the network shape.
        The last element in each individual vector stores the network's fitness. '''
    population = []
    size = 0
    for i in range(len(shape)-1):
        size += shape[i] * shape[i+1]
    for i in range(mu):
        population.append(np.random.uniform(0, 0.1, size+1))
        #population[i][-1] = 0
    return population

def evaluate(shape, population, data, quick=False):
    ''' Tests the supplied data on each individual in the population, storing the overall
            fitness of the network as the last element of each individual's array. '''
    if quick: data = random.sample(data, int(len(data)/3))
    for individual in population:
        nn = build_network(shape, individual)
        individual[-1] = nn.test(data)

def build_network(shape, individual):
    ''' Builds a neural net of the input shape using the individual's array of weights. '''
    nn = NN.Network(shape)
    counter = 0
    for layer in nn.weights:
        for node in layer:
            for i in range(len(node)):
                node[i] = individual[counter]
                counter += 1
    return nn

def tournament_unique(population, number_possible_parents, number_parents):
    ''' Returns unique parents of parents from the population using tournament selection.
        Returns , i.e. no parents can be selected more than once.'''
    parents = []
    for i in range(number_parents):
        parents.append(sorted(random.sample(population, number_possible_parents), key=itemgetter(-1))[0])
    return np.unique(parents, axis=0)


def reproduce(parents, crossover_rate):
    ''' Creates children from each combination of supplied parents, preventing duplication
        (i.e. the same parent can't mate with itself). '''
    children = []
    for parent1 in parents:
        for parent2 in parents:
            if not (parent1 == parent2).all():
                child1, child2 = crossover(parent1, parent2, crossover_rate)
                children.append(child1)
                children.append(child2)
    return children

def crossover(parent1, parent2, crossover_rate):
    ''' Performs uniform crossover on the supplied parents. '''
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1) - 1):
        if random.random() < crossover_rate:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    return child1, child2

def mutate(child, mutation_rate):
    ''' Mutates a child, each gene has an equal chance of mutation. Can be pushed in the positive or negative direction. '''
    for i in range(len(child)-1):
        if np.random.random() < mutation_rate:
            child[i] += np.random.random() - 0.5

def train(shape, mu, generations, number_possible_parents, number_parents, data, print_frequency = 250, target_fitness = 0.0001, mutation_rate = 0.2, crossover_rate = 0.7, quick=False):
    ''' Trains a network using the genetic algorithm. Returns the best network created after either the max
        number of generations have been run, or the target_fitness has been achieved. '''

    lifetime_error = ['Genetic Algorithm {}'.format(shape)]

    # create and evaluate the original seed population
    population = create_population(shape, mu)
    evaluate(shape, population, data, quick)


    for i in range(generations+1):

        parents = tournament_unique(population, number_possible_parents, number_parents)
        #print("Tournament selected {} parents".format(number_parents))
        children = reproduce(parents, crossover_rate)
        #print("Parents reproduced {} children".format(len(children)))

        for child in children:
            #print("Child before mutation = {}".format(child))
            mutate(child, mutation_rate)
            #print("Child after mutation = {}".format(child))

        evaluate(shape, children, data, quick)

        # keeps the best individuals from the combined children and population arrays to act as the
        # next generations population
        if len(children) != 0: population = np.concatenate((population, children))
        population = sorted(population, key=itemgetter(-1))[:mu] # elitist selection
        error = [individual[-1] for individual in population]
        lifetime_error.append(statistics.mean(error))
        #lifetime_error.append(population[0][-1])

        if population[0][-1] < target_fitness: break

        if i%print_frequency == 0:
            print("Generation {}'s fitnesses are: ".format(i), end='')
            for i in population:
                print("{}, ".format(i[-1]), end = '')
            print()

    return build_network(shape, population[0]), lifetime_error


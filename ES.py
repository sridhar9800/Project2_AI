'''
Prashant Sridhar (54565839)
Simple Experimentation.

This python module is used to train a neural network using the (mu + lambda) Evolution Strategy. Uses crossover,
create_population, evaluation, tournament_unique, and reproduce functions from GA.
'''

import GA
import numpy as np
from operator import itemgetter
import math
import statistics

def create_es_population(shape, mu):
    population = GA.create_population(shape, mu)
    for i in range(len(population)):
        individual = population[i][:-1]
        individual = np.concatenate((individual, np.random.normal(0, 1, len(individual)+1)))
        population[i] = individual
        #population[i][-1] = 0
    return population


def mutate(child, mutation_rate):
    ''' Mutates a supplied child. Each value in the vector has an equal chance of being mutated. '''
    offset = int((len(child)-1)/2)
    for i in range(offset):
        if np.random.random() < mutation_rate:
            # first mutate the associated sigma value
            child[i + offset] = abs(child[i + offset] * math.exp(np.random.normal(0, 1) / math.sqrt((len(child) - 1) / 2)))
            # then mutate the actual value of the child
            child[i] = child[i] + np.random.normal(0, child[i + offset])


def train(shape, mu, generations, number_possible_parents, number_parents, data, print_frequency = 250, target_fitness = 0.00001, mutation_rate = 0.5, crossover_rate = 0.5, quick=False):
    ''' Trains a network using the mu + lambda evolution strategy. Returns the best network created after either the max
        number of generations have been run, or the target_fitness has been achieved. '''

    lifetime_error = ['Evolution Strategy {}'.format(shape)]

    # create and evaluate the original seed population
    population = create_es_population(shape, mu)
    GA.evaluate(shape, population, data, quick)

    for i in range(generations+1):

        #parents = GA.tournament(population, number_possible_parents, number_parents)
        parents = GA.tournament_unique(population, number_possible_parents, number_parents)
        #print("Tournament selected {} parents".format(number_parents))
        children = GA.reproduce(parents, crossover_rate)
        #print("Parents reproduced {} children".format(len(children)))

        for child in children:
            #print("Child before mutation = {}".format(child))
            mutate(child, mutation_rate)
            #print("Child after mutation = {}".format(child))

        GA.evaluate(shape, children, data, quick)

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

    return GA.build_network(shape, population[0]), lifetime_error

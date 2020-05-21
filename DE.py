'''
Prashant Sridhar (54565839)
Simple Experimentation.

This python module is used to train a neural network using the Genetic Algorithm. Some functionality is
reused by other algorithms.
'''

import GA
import random
import copy
from operator import itemgetter

def train(shape, mu, generations, data, print_frequency=5, target_fitness=0.0001, crossover_rate=0.5, weight=0.5):
    ''' Trains a network using Differential Evolution. Returns the best network created after either the max
            number of generations have been run, or the target_fitness has been achieved. '''

    population = GA.create_population(shape, mu)
    fitness = 100
    lifetime_error = ['Differential Evolution {}'.format(shape)]
    #print(len(population))
    GA.evaluate(shape, population, data)

    for gen in range(generations+1):

        for j in range(len(population)):
            individuals = random.sample(range(len(population)), 4)
            cross = random.randint(0, len(population[0]))
            temp = copy.deepcopy(population[individuals[0]])
            for i in range(len(population[0]) - 1):
                if (random.random() < crossover_rate or cross == i):
                    temp[i] = population[individuals[1]][i] + (
                    weight * (population[individuals[2]][i] - population[individuals[3]][i]))
            GA.evaluate(shape, [temp, population[individuals[0]]], data)
            if (temp[-1] < population[individuals[0]][-1]):
                population[individuals[0]] = temp
            fitness = population[individuals[0]][-1]
        population = sorted(population, key=itemgetter(-1))[:mu]  # elitist selection
        lifetime_error.append(population[0][-1])

        if population[0][-1] < target_fitness: break

        if gen % print_frequency == 0:
            print("Generation {}'s fitnesses are: ".format(gen), end='')
            for k in population:
                print("{}, ".format(k[-1]), end='')
            print()

    return GA.build_network(shape, population[0]), lifetime_error

import fitness_func
import repair_population
import find_neighborhood
import compute_mean_performance_mean_gens
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import test_eval_algorithm
import find_neighborhood
import os

from deap import base
from deap import creator
from deap import tools

# creating the user-movie matrix
pwd = os.getcwd()+'\\'+'ua.base'
data = pd.read_csv(pwd, sep='\t',
                   names=['user', 'item', 'rating', 'timestamp'], header=None, encoding='utf-8')
data = data.drop('timestamp', axis=1)

# obtaning user-movie matrix from the movielens data
matrix = data.pivot(index='user', columns='item', values='rating')

new_index_column = np.arange(1682) + 1
matrix = matrix.reindex(new_index_column, axis=1)


matrix = matrix.fillna(0)
user_movie_matrix = matrix.to_numpy()

index_number_of_the_user_under_cons = 138

user_under_consideration = user_movie_matrix[ index_number_of_the_user_under_cons , :]


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("bird", random.randint, 1, 5)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 1682(number of movie ratings)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.bird, 1682)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


neighborhood = find_neighborhood.get_neighborhood(user_movie_matrix,index_number_of_the_user_under_cons)
#fitness function
def evaluate_ind(individual,neigh):
    m = np.mean(individual)
    mean_vect = np.tile(m,len(individual))
    individual_centered = individual - mean_vect

    norm2 = np.sqrt(individual_centered.dot(individual_centered))
    individual_centered_normalized = individual_centered/norm2
    #---------------------------------------------

    correlations = neigh.dot(individual_centered_normalized)

    #scale the correlations [-1,1] -> [0,2] in order to take non negative
    correlations = correlations + 1

    return np.sum(correlations),



# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evaluate_ind , neigh = neighborhood )

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint   )

# register a mutation operator with a probability to
# flip each attribute/gene of 0.07
toolbox.register("mutate", tools.mutUniformInt, low = 1 , up = 5 , indpb=0.07)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.

toolbox.register("select", tools.selTournament , tournsize = 10 , fit_attr='fitness' )
#toolbox.register("select", tools.selRoulette , fit_attr='fitness' )

# ----------

best_individuals_for_each_run = {}
h = []

c = []
for k in range(10):
    random.seed()

    number_of_individual_per_generation = 20
    number_of_generations = 150

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=number_of_individual_per_generation)


    pop_repaired = repair_population.repair(pop,user_under_consideration)
    pop[:] = pop_repaired


    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.6 , 0.1

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    #Variable keeping track of the number of generations the best individual stopped increasing its fitness
    counter = 0;

    p = 0;


    best_ind_for_each_gen = []
    l = tools.selBest(pop, 1)[0]
    best_fitness = l.fitness.values[0]
    best_ind_for_each_gen.append(best_fitness)
    h.append(l)

    # Begin the evolution
    while  (g < number_of_generations) :
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)



        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                #del child1.fitness.values
                #del child2.fitness.values


        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                #del mutant.fitness.values
          


        '''
        #elitism
        while p < len(pop)-1:
            p = p + 1;
            if random.random() < MUTPB:
                offspring[p] = l
        p = 0;
        '''



        #fixing posible non valid solutions and updating the new generation in order to jump to the next iteration
        offspring = repair_population.repair(offspring,user_under_consideration)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Evaluate the entire population and assigning the fitnesses to each corresponding individual of the pop
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit


        #selecting the best individual of each generation
        l = tools.selBest(pop, 1)[0]
        best_fitness = l.fitness.values[0]
        #best_fitness = evaluate_ind(l,neighborhood)
        best_ind_for_each_gen.append(best_fitness)
        h.append(l)


        '''
        #early stopping logic
        if best_ind_for_each_gen[g] <= best_ind_for_each_gen[g-1] :
            counter = counter + 1
        else :
            counter = 0

        if counter > 20 :
            break
        '''


        '''
        if ( (best_ind_for_each_gen[g]-best_ind_for_each_gen[g-1])*100 )  < 1 :
            break
        '''
    c.append(best_ind_for_each_gen)
    best_individuals_for_each_run[k] = h
    h = []



#finding the mean individuals per generation for the 10 runs---------------------
o = []
for i in best_individuals_for_each_run:
    o.append(best_individuals_for_each_run[i])

#o is gonna be a tensor with dims (10,201,1682) (run,generation,rating)
o = np.array(o)
#obtaining the mean individuals
Best_mean_individuals = np.rint( np.mean( np.array(o) , axis=0 ) )

test_eval_algorithm.testing(Best_mean_individuals,index_number_of_the_user_under_cons)
#---------------------------------------------------------------------------------


#plotting the fitness of the mean best individual per generation for the 10 runs of the algorithm
best_ind_for_each_gen = np.mean(c,axis=0)
plt.figure()
plt.plot(np.arange(g+1),best_ind_for_each_gen)
plt.xlabel('generation')
plt.ylabel('best individual fitness (mean for each run)')
plt.show()

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

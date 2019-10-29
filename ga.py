import numpy as np
import GlobalIndexKinematical as km

def fitness(equation_inputs, pop):
    # 
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def select_parents(pop, fitness, num_parents):
    parents = np.zeros([num_parents,pop.shape[1]])
    for parent_num in range(num_parents):
        max_fit_index = np.where(fitness == np.max(fitness))
        max_fit_index = max_fit_index[0][0]
        parents[parent_num,:] = pop[max_fit_index,:]
        fitness[max_fit_index] = -999999999
    return parents


def crossover(parents,num_children):
    children = np.zeros([num_children,parents.shape[1]])
    crossover_point = np.int(parents.shape[1]/2)
    for child in range(num_children):  
        parent1_index = child%parents.shape[0]
        parent2_index = (child+1)%parents.shape[0]
        if (child<num_children/2):  
            children[child, 0:crossover_point] = parents[parent1_index,0:crossover_point]
            children[child, crossover_point:] = parents[parent2_index,crossover_point:]
        else:
            children[child, 0:crossover_point] = parents[parent2_index,0:crossover_point]
            children[child, crossover_point:] = parents[parent1_index,crossover_point:]

    return children

def mutation(offspring_cross,span,threshold):
    #offspring_cross = np.concatenate((offspring_cross,offspring_cross))
    for child in range(offspring_cross.shape[0]):
        p = np.random.uniform(0,1,1)
        if p < threshold :
            k = np.random.randint(0, offspring_cross.shape[1])
            print('Mutation on child: ', child,' - Gen: ', k, '\n')
            mut_value = np.random.uniform(span[0][k], span[1][k], 1)
            offspring_cross[child,k] = mut_value
    
    return offspring_cross

def fitnessK(population):
    fitness_population = []
    for L in population:
        I = km.AllIndex(L)
        I = km.IntegratedIndex(I)
        I = km.GlobalIndex(I)
        fitness_population.append(I)
    fitness_population = np.matrix(fitness_population)
    return fitness_population




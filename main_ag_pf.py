import numpy as np
import ga

# Inputs or constants
equation_inputs = [4,-2,3.5,-11]
span = [[0,0,0,0],[11,12,13,14]]


# Number of variables for optimization
num_var = 4
num_kromo = 10
pop_size = (num_kromo, 1)

new_pop1 = np.random.uniform(low=0, high=11, size=pop_size)
new_pop2 = np.random.uniform(low=0, high=12, size=pop_size)
new_pop3 = np.random.uniform(low=0, high=13, size=pop_size)
new_pop4 = np.random.uniform(low=0, high=14, size=pop_size)

new_population = np.concatenate((new_pop1,new_pop2, new_pop3, new_pop4),axis=1)
print('New population:\n',new_population,'\n')

# Number of Parents
num_parents = int(num_kromo/2)
k = 1 # Number of Generations

# Evaluate Fitness
fitness = ga.fitness(equation_inputs, new_population)
print('Fitness:\n',fitness.transpose(),'\n')

# Select Best Fitness
parents = ga.select_parents(new_population,fitness,num_parents)
print('Parents Selected:\n',parents, '\n')

# 
offspring_cross = ga.crossover(parents,new_population.shape[0])
print('crossover:\n',offspring_cross, '\n')

# Mutation
mut_prob = 0.3
offspring_mut = ga.mutation(offspring_cross,span, mut_prob)
print('Mutation:\n',offspring_mut, '\n')

# Pop
generation = np.concatenate((parents, offspring_mut))
print('Generation ', k, ':\n', generation)
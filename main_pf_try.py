import numpy as np
import matplotlib.pyplot as plt
import ga
import GlobalIndexKinematical as km

# Inputs or constants
span = [[500,50,50,-100],[1000,250,250,100]]


# Number of variables for optimization
num_var = 4
num_kromo = 5
pop_size = (num_kromo, 1)

new_pop1 = np.random.uniform(span[0][0], span[1][0], size=pop_size)
new_pop2 = np.random.uniform(span[0][1], span[1][1], size=pop_size)
new_pop3 = np.random.uniform(span[0][2], span[1][2], size=pop_size)
new_pop4 = np.random.uniform(span[0][3], span[1][3], size=pop_size)

new_population = np.concatenate((new_pop1,new_pop2, new_pop3, new_pop4),axis=1)
print('Random Population:\n',new_population,'\n')

# Point Cloud
P = km.WorkspaceDesired(500.0,650.0,50.0)

# Number of Parents
num_parents = int(num_kromo/2)
k = 10 # Number of Generations

Global_fitness = []

for i in range(k):
    # Evaluate Fitness
    print('\n',f'Evaluate Fitness of Generation {i}:')
    fitness = ga.fitnessK(new_population,P)
    Global_fitness.append(max(fitness))
    #print('Fitness:\n',fitness,'\n')

    # Select Best Fitness
    parents = ga.select_parents(new_population,fitness,num_parents)
    print('Parents Selected:\n',parents, '\n')

    # Crossover
    offspring_cross = ga.crossover(parents,num_kromo)
    print('crossover:\n',offspring_cross, '\n')

    # Mutation
    mut_prob = 0.4
    offspring_mut = ga.mutation(offspring_cross,span, mut_prob)
    print('Mutation:\n',offspring_mut, '\n')

    # Pop
    new_population = np.concatenate((parents, offspring_mut))
    print('Generation ', i, ':\n', new_population)

print('\n',f'Evaluate Fitness of Generation {k}:')
fitness = ga.fitnessK(new_population,P)
Global_fitness.append(max(fitness))

#print(Global_fitness)

plt.plot(Global_fitness,'b-')

plt.xlabel('# Generaciones')
plt.show()
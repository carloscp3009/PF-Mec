import numpy as np
import matplotlib.pyplot as plt
import ga

# Inputs or constants
equation_inputs = [4,-2,3.5,-11]
span = [[0,0,0,0],[9,9,9,9]]


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


# Number of Parents
num_parents = int(num_kromo/2)
k = 2 # Number of Generations

Global_fitness = []

for i in range(k):
    # Evaluate Fitness
    fitness = ga.fitnessK(new_population)
    Global_fitness.append(max(fitness))
    print('Fitness:\n',fitness.transpose(),'\n')

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

#print(Global_fitness)

plt.plot(Global_fitness,'b--')

plt.xlabel('# Generaciones')
plt.show()
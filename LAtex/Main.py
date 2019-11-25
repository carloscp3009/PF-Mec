import numpy as np
import matplotlib.pyplot as plt
import ga
import GlobalIndexKinematical as km


# Variables Range
span = [[500,50,50,-100],[1000,250,250,100]]

# Number of variables for optimization
num_var = 4
num_kromo = 10
pop_size = (num_kromo, 1)

new_pop1 = np.random.uniform(span[0][0], span[1][0], size=pop_size)
new_pop2 = np.random.uniform(span[0][1], span[1][1], size=pop_size)
new_pop3 = np.random.uniform(span[0][2], span[1][2], size=pop_size)
new_pop4 = np.random.uniform(span[0][3], span[1][3], size=pop_size)
new_population = np.concatenate((new_pop1,new_pop2, new_pop3, new_pop4),axis=1)

# Point Cloud - WorkSpace
P = km.WorkspaceDesired(500.0,650.0,50.0)

# Number of Parents
num_parents = int(num_kromo/2)
k = 120 # Number of Generations

Global_fitness = []
Avg_fitness=[]
Minor_fitness=[]
Mf_chromo=[]

for i in range(k):
    # Evaluate Fitness
    fitness = ga.fitnessK(new_population,P)
    Global_fitness.append(max(fitness))

    # Average Fitness
    Avg_fitness.append(sum(fitness)/len(fitness))

    # Select Best Fitness
    parents = ga.select_parents(new_population,fitness,num_parents)

    # Crossover
    offspring_cross = ga.crossover(parents,num_kromo)

    # Mutation
    mut_prob = 0.6
    offspring_mut = ga.mutation(offspring_cross,span, mut_prob)

    # Pop
    new_population = np.concatenate((parents, offspring_mut))

print('\n','Evaluate Fitness of Generation',i,': ')
fitness = ga.fitnessK(new_population,P)
Global_fitness.append(max(fitness))

# Best Child
winner_chromo = new_population[fitness.index(max(fitness))]
print('Best Chromo: ',winner_chromo)

win_local_idx = km.AllIndex(winner_chromo,P)
np.savetxt('win_local_idxs.csv', win_local_idx, delimiter = ',')
np.savetxt('Global_idx.csv',Global_fitness, delimiter = ',')

plt.plot(Global_fitness,'b-')
plt.plot(Avg_fitness,'r-')

plt.xlabel('# Generaciones')
plt.show()



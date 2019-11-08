import numpy as np
import matplotlib.pyplot as plt
import ga
import GlobalIndexKinematical as km
from mpl_toolkits.mplot3d import Axes3D

P = km.WorkspaceDesired(500.0,650.0,50.0)
idx = range(len(P[:,0]))

#idx = range(len(P[0:500,0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = P[:,0]
y = P[:,1] 
z = P[:,2]
# x = P[0:500,0]
# y = P[0:500,1] 
# z = P[0:500,2]

colors=plt.cm.jet(idx)
print(type(colors))
# colors=plt.cm.jet(idx[0:500])

plt.style.context(('ggplot')) 
ax.scatter(x, y, z,c=colors, s=50)

plt.show()
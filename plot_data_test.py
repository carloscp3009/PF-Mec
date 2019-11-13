import numpy as np
import matplotlib.pyplot as plt
import ga
import GlobalIndexKinematical as km
from mpl_toolkits.mplot3d import Axes3D

P = km.WorkspaceDesired(500.0,650.0,50.0)
local_idx = np.genfromtxt('local_idxs.csv', delimiter=',')
idx1 = local_idx[:,0]
idx1 = idx1/max(idx1)
idx2 = local_idx[:,1]
idx2 = idx2/max(idx2)
idx3 = local_idx[:,2]
idx3 = idx3/max(idx3)
idx4 = local_idx[:,3]
idx4 = idx4/max(idx4)

fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

x = P[:,0]
y = P[:,1] 
z = P[:,2]
colors1=plt.cm.jet(idx1)
colors2=plt.cm.jet(idx2)
colors3=plt.cm.jet(idx3)
colors4=plt.cm.jet(idx4)

plt.style.context(('ggplot')) 
ax1.scatter(x, y, z,c=colors1, s=20)
ax2.scatter(x, y, z,c=colors2, s=20)
ax3.scatter(x, y, z,c=colors3, s=20)
ax4.scatter(x, y, z,c=colors4, s=20)

plt.show()
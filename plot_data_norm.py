import numpy as np
import matplotlib.pyplot as plt
import ga
import GlobalIndexKinematical as km
from mpl_toolkits.mplot3d import Axes3D

P = km.WorkspaceDesired(500.0,650.0,50.0)
x = P[:,0]
y = P[:,1] 
z = P[:,2]

lose_local_idx = np.genfromtxt('lose_local_idxs.csv', delimiter=',')
win_local_idx = np.genfromtxt('win_local_idxs.csv', delimiter=',')

idx1 = lose_local_idx[:,0]
idx2 = lose_local_idx[:,1]
idx3 = lose_local_idx[:,2]
idx4 = lose_local_idx[:,3]
idx5 = win_local_idx[:,0]
idx6 = win_local_idx[:,1]
idx7 = win_local_idx[:,2]
idx8 = win_local_idx[:,3]
idx1 = 1-(idx1-min(idx1))/(max(idx5)-min(idx1))
idx2 = 1-(idx2-min(idx2))/(max(idx6)-min(idx2))
idx3 = 1-(idx3-min(idx3))/(max(idx7)-min(idx3))
idx4 = 1-(idx4-min(idx4))/(max(idx8)-min(idx4))
idx5 = 1-(idx5-min(idx1))/(max(idx5)-min(idx1))
idx6 = 1-(idx6-min(idx2))/(max(idx6)-min(idx2))
idx7 = 1-(idx7-min(idx3))/(max(idx7)-min(idx3))
idx8 = 1-(idx8-min(idx4))/(max(idx8)-min(idx4))

print('max: ', max(idx1),', min: ', min(idx1))
print('max: ', max(idx2),', min: ', min(idx2))
print('max: ', max(idx3),', min: ', min(idx3))
print('max: ', max(idx4),', min: ', min(idx4))
print('max: ', max(idx5),', min: ', min(idx5))
print('max: ', max(idx6),', min: ', min(idx6))
print('max: ', max(idx7),', min: ', min(idx7))
print('max: ', max(idx8),', min: ', min(idx8))


fig = plt.figure()
ax1 = fig.add_subplot(241, projection='3d')
ax2 = fig.add_subplot(242, projection='3d')
ax3 = fig.add_subplot(243, projection='3d')
ax4 = fig.add_subplot(244, projection='3d')
ax5 = fig.add_subplot(245, projection='3d')
ax6 = fig.add_subplot(246, projection='3d')
ax7 = fig.add_subplot(247, projection='3d')
ax8 = fig.add_subplot(248, projection='3d')

colors1=plt.cm.jet(idx1)
colors2=plt.cm.jet(idx2)
colors3=plt.cm.jet(idx3)
colors4=plt.cm.jet(idx4)
colors5=plt.cm.jet(idx5)
colors6=plt.cm.jet(idx6)
colors7=plt.cm.jet(idx7)
colors8=plt.cm.jet(idx8)

plt.style.context(('ggplot')) 
ax1.scatter(x, y, z,c=colors1, s=20, alpha=0.7)
ax2.scatter(x, y, z,c=colors1, s=20, alpha=0.7)
ax3.scatter(x, y, z,c=colors3, s=20, alpha=0.7)
ax4.scatter(x, y, z,c=colors4, s=20, alpha=0.7)
ax5.scatter(x, y, z,c=colors5, s=20, alpha=0.7)
ax6.scatter(x, y, z,c=colors6, s=20, alpha=0.7)
ax7.scatter(x, y, z,c=colors7, s=20, alpha=0.7)
ax8.scatter(x, y, z,c=colors8, s=20, alpha=0.7)

ax1.set_title('Mr inicial')
ax2.set_title('Vm inicial')
ax3.set_title('\u03BCr inicial')
ax4.set_title('Kj inicial')
ax5.set_title('Mr optimizado')
ax6.set_title('Vm optimizado')
ax7.set_title('\u03BCr optimizado')
ax8.set_title('Kj optimizado')

# frame1 = plt.gca()
# frame1.axes.get_xaxis().set_ticks([])
# frame1.axes.get_yaxis().set_ticks([])

ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.yaxis.set_major_formatter(plt.NullFormatter())
ax1.zaxis.set_major_formatter(plt.NullFormatter())

ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax2.zaxis.set_major_formatter(plt.NullFormatter())

ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax3.yaxis.set_major_formatter(plt.NullFormatter())
ax3.zaxis.set_major_formatter(plt.NullFormatter())

ax4.xaxis.set_major_formatter(plt.NullFormatter())
ax4.yaxis.set_major_formatter(plt.NullFormatter())
ax4.zaxis.set_major_formatter(plt.NullFormatter())

ax5.xaxis.set_major_formatter(plt.NullFormatter())
ax5.yaxis.set_major_formatter(plt.NullFormatter())
ax5.zaxis.set_major_formatter(plt.NullFormatter())

ax6.xaxis.set_major_formatter(plt.NullFormatter())
ax6.yaxis.set_major_formatter(plt.NullFormatter())
ax6.zaxis.set_major_formatter(plt.NullFormatter())

ax7.xaxis.set_major_formatter(plt.NullFormatter())
ax7.yaxis.set_major_formatter(plt.NullFormatter())
ax7.zaxis.set_major_formatter(plt.NullFormatter())

ax8.xaxis.set_major_formatter(plt.NullFormatter())
ax8.yaxis.set_major_formatter(plt.NullFormatter())
ax8.zaxis.set_major_formatter(plt.NullFormatter())

plt.show()
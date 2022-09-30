import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


maxd = 100
maxn = 400
dvec = np.arange(10, maxd + 1, 10)
nvec = np.arange(10, maxn + 1, 10)
save_folder = './results/'
fname = '' # put the file name here
dis = np.load(save_folder + 'irr1_' + fname + '.npy')
dis = np.mean(dis, axis=2)
X, Y = np.meshgrid(nvec, dvec)
Z = dis.T
fig, ax = plt.subplots()
# if plot phase transition for distance, use extend='max'
cs = ax.contourf(X, Y, Z, levels=np.arange(0,1.1,0.1), cmap=cm.jet, extend='max')
# if plot phase transition for probability
# cs = ax.contourf(X, Y, Z, levels=np.arange(0,1.1,0.1), cmap=cm.jet)
# if plot the boundary of success with probability 1
# cs2 = ax.contour(X, Y, Z, levels=[0.9, 1], colors=('k',), linewidths=(2,))
fig.colorbar(cs, ax=ax)
ax.set_xlabel('n')
ax.set_ylabel('d')
plt.show()
fig.savefig(save_folder + fname + '.png')

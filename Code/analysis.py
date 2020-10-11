from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

lep_rep = np.load('leg_rep.npy')
voter_rep = np.load('voter_rep.npy')
label = np.load('label.npy')
all_rep = np.concatenate([lep_rep[0].reshape([1,32]),voter_rep],axis=0)
all_label = np.concatenate([[4],label])
pca = PCA(n_components=2)
pca_rep = pca.fit_transform(all_rep)


import numpy as np
import matplotlib.pyplot as plt

volume = np.ones(len(all_label))
volume[0]=2

fig, ax = plt.subplots()
ax.scatter(pca_rep[:,0], pca_rep[:,1], c=label, s=volume, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and percent change')

ax.grid(True)
fig.tight_layout()

plt.show()
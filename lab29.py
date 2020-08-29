import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(2000, 2) + [2, 2],
          np.random.randn(2000, 2) + [0, -2],
          np.random.randn(2000, 2) + [-2, 2]]

kmeans = KMeans(n_clusters=3,
                n_init=10,
                max_iter=300)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
colors = ['c', 'm', 'y', 'k']
markers = ['o', 'v', '*', 'x']
for i in range(3):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1],
                c=colors[i], marker=markers[i])
    print(f'group{i} has {dataX.size}')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='*',
            s=200,
            c='#0599FF')
plt.show()

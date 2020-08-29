import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(2000, 2) + [2, 2],
          np.random.randn(2000, 2) + [0, -2],
          np.random.randn(2000, 2) + [-2, 2]]

interias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    interias.append(kmeans.inertia_)
print(interias)
plt.plot(range(1,10), interias)
plt.show()
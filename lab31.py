import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])

shortestNeighbors = NearestNeighbors(n_neighbors=2,
                                     algorithm='auto').fit(X)
distances, indices = \
    shortestNeighbors.kneighbors(X, return_distance=True)
print(distances)
print(indices)
print(shortestNeighbors.kneighbors_graph(X).toarray())
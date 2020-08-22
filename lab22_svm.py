import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
classifier1 = SVC()
classifier1.fit(X, y)
print(classifier1)

print("predict=", classifier1.predict([[1, 0], [0, 1],
                                       [0.5, 0.5], [0.5, -0.5],
                                       [-1, 0], [0, -1]]))

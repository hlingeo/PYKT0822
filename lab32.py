
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
Y = [1, 1, 1, 2, 2, 2]
classifier1 = GaussianNB()
classifier1.fit(X, Y)
print(classifier1.predict([[-0.8,-0.8],[2.1, 2.5]]))

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
print(classifier2.predict([[-0.4, -0.4]]))
classifier2.partial_fit([[-0.5, -0.5]], [2])
print(classifier2.predict([[-0.4, -4]]))



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

data1 = datasets.make_regression(100, 1, noise=5)
print(type(data1))
# (100,1) for x->2D, (100,) for y->1D
print(data1[0].shape, data1[1].shape)
#1D
print(np.array([1, 3, 5, 7]).shape)
#2D
print(np.array([[1], [3], [5], [6]]).shape)
plt.scatter(data1[0], data1[1], c='green', marker='^')
plt.show() 
regression1 = linear_model.LinearRegression()
regression1.fit(data1[0],data1[1])
print(f'coef ={regression1.coef_}, intercept={regression1.intercept_}')
print(f"regression score = {regression1.score(data1[0], data1[1])}")
range1 = [-3, 3]
#a1*x1+b
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='blue')
plt.scatter(data1[0], data1[1],c='green', marker='^')
plt.show()

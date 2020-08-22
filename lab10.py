import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
plt.plot(x, y,'r--')
plt.scatter(x, y)
plt.show()
regression1 = LinearRegression()
regression1.fit(x,y)
x_seq = np.array(np.arange(5, 55, 0.1)).reshape(-1, 1)
plt.plot(x, y, 'r--')
plt.scatter(x, y)
plt.plot(x, regression1.coef_ * x + regression1.intercept_, 'g-')
plt.show()
print(f"1st order linear regression score={regression1.score(x, y)}")

transformer = PolynomialFeatures (degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
print(f"x shape={x.shape}, x_ shape={x_.shape}")
print(x)
print(x_)
regression2 = LinearRegression().fit(x_, y)
print(f"2nd order linear regression score={regression2.score(x_, y)}")



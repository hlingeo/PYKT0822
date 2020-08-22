import matplotlib.pyplot as plt
from sklearn import linear_model

# first x1,x2 second, x1, x2, and so on
features = [[0, 1], [1, 3], [2, 8]]
labels = [1, 4, 5.5]
regression1 = linear_model.LinearRegression()
regression1.fit(features, labels)
print(f"coef={regression1.coef_}")
print(f"intercept={regression1.intercept_}")
# y= a1x1+a2x2
print(f"a1={regression1.coef_[0]}, a2={regression1.coef_[1]}")
newpoints = [[0.2, 0.8], [1, 2],
             [0.8, 0.8], [2, 1], [10, 14]]
guess = regression1.predict(newpoints)
print(guess)
mapping = [1.9, 3.25, 4.3, 9., 34.5]
print(regression1.score(newpoints, mapping))
real = [2, 4, 5, 10, 35]
print(regression1.score(newpoints, real))
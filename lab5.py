import matplotlib.pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
features = [[1], [2], [3], [4]]
labels = [1, 4, 15, 18]
plt.scatter(features, labels, c='green')
plt.show()
regression1.fit(features,labels)
print(f'coef={regression1.coef_}')
print(f"intercept = {regression1.intercept_}")
range1 = [0,4]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='gray')
plt.scatter(features, labels, c='green')
plt.show()

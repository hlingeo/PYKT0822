from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

x, y = make_regression(n_samples=100, n_features=10,
                       n_informative=5)
model1 = LinearRegression()
model1.fit(x, y)
# print(x)
importance = model1.coef_
print(importance)
for i, v in enumerate(importance):
    print("feature:%0d, score:%.5f" % (i, v))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

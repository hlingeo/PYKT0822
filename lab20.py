from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))

print(iris.feature_names[3])
X = iris["data"][:, 3:]
print(iris.target_names[2])
y = (iris["target"] == 2).astype(np.int)

logistic_regression1 = LogisticRegression()
logistic_regression1.fit(X, y)
print(logistic_regression1.coef_, logistic_regression1.intercept_)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = logistic_regression1.predict_proba(X_new)
# get parameters from logistic regression
a = logistic_regression1.coef_[0]
b = logistic_regression1.intercept_
my_plot = 1 / (1 + np.exp(-(a * X_new + b)))
plt.plot(X, y, "g.")
plt.plot(X_new, y_prob[:, 1], "r-", label="Iris_Virginica")
plt.plot(X_new, y_prob[:, 0], "b--", label='Not Iris_Virginica')
plt.plot(X_new, my_plot, label="formula")
plt.grid()
plt.legend()
plt.xlabel("petal width")
plt.ylabel("probability")
plt.show()




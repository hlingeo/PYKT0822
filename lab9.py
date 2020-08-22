from sklearn import datasets
import matplotlib.pyplot as plt

data1 = datasets.make_regression(10, 6, noise=5)
for i in range(0, data1[0].shape[1]):
    # where there will be 5 data set
    x = data1[0][:, i]
    y = data1[1]
    plt.title(f"#{i} variable")
    plt.scatter(x, y)
    plt.show()
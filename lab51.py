from pandas import read_csv
import numpy as np

df1 = read_csv("./data/iris.data", header=None)
dataset = df1.values
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features.shape, labels.shape)
print(np.unique(labels, return_counts=True))

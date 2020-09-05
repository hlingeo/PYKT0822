import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(numpy.unique(y, return_counts=True))
print(len((numpy.unique(numpy.hstack(X)))))
lengths = [len(x) for x in X]
print(numpy.mean(lengths), numpy.std(lengths))

plt.subplot(121)
plt.boxplot(lengths)
plt.subplot(122)
plt.hist(lengths)
plt.show()

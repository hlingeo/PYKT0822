import tensorflow as tf
import numpy as np
import tensorflow as tf

scores = [3.0, 1.0, 2.0]


def my_softmax(x):
    ax = np.array(x)
    return np.exp(ax) / np.sum(np.exp(ax), axis=0)


def normal_ratio(x):
    ax = np.array(x)
    return ax / np.sum(ax, axis=0)

print(my_softmax(scores), normal_ratio(scores))
print(tf.nn.softmax(scores).numpy())
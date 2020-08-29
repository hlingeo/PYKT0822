import tensorflow as tf

vectors = [3, -1, 2.4, 5.9, 0.0001, 0.51, 0.49, -0.0008]

result1 = tf.nn.relu(vectors)
print("after apply relu, result1=", result1)
result2 = tf.nn.sigmoid(vectors)
print("after apply relu, result2=", result2)
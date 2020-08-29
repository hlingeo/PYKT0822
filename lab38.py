import tensorflow as tf
# what would happn in tensorflow v1

tf.compat.v1.disable_eager_execution()
hello1 = tf.constant('Hello Tensorflow!')
session1 = tf.compat.v1.Session()
print(hello1)
print(session1.run(hello1)) # in v1, you have to run the constant to show you the value
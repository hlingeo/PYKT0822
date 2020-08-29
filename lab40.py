import tensorflow as tf
# tf v1
tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)

with tf.compat.v1.Session() as session:
    result = session.run(c, feed_dict={
        a: [3, 4, 5],
        b: [-1, -2, -3]
    })
    print(result)



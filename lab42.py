import tensorflow as tf
from datetime import datetime


@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (tf.add(tf.add(a, b), c)) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


stamp = datetime.now().strftime("%Y%m%d-%H%M")
logdir = 'logs/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on(graph=True, profiler=True)
print(computeArea(tf.constant([[5.0, 3.0, 4.0],
                               [6.0, 6.0, 6.0]])))
with writer.as_default():
    tf.summary.trace_export(name='lab42',
                            step=0,
                            profiler_outdir=logdir)
tf.summary.trace_off()
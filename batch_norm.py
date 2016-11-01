import tensorflow as tf
import numpy as np

is_training = tf.Variable(True)
def bn(x):
  shape = x.get_shape().as_list()

  beta = tf.get_variable('beta', shape[-1:],
                         initializer=tf.zeros_initializer)
  gamma = tf.get_variable('gamma', shape[-1:],
                          initializer=tf.ones_initializer)

  mean = tf.reduce_mean(x, [0])
  std = tf.sqrt(tf.reduce_mean((x - mean)**2, [0]))

  ema = tf.train.ExponentialMovingAverage(decay=0.5)
  tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                       ema.apply([mean, std]))

  mean, std = tf.cond(is_training,
                      lambda: (mean, std),
                      lambda: (ema.average(mean), ema.average(std)))
  return gamma*(x-mean)/std + beta


if __name__ == '__main__':
  a = 10.0*np.random.random((64, 32, 32, 3))
  x = tf.placeholder(tf.float32, [None, 32 , 32, 3])
  x_out = bn(x)
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  x_val, _ = sess.run([x_out, tf.moving_average_variables()], {x: a})
  print x_val.mean(), np.std(x_val)
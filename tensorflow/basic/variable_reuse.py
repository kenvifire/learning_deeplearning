import tensorflow as tf

session = tf.InteractiveSession()


def not_so_simple_computation(w):
    x = tf.get_variable('x', initializer=tf.constant(5.0, shape=None, dtype=tf.float32))
    y = tf.get_variable('y', initializer=tf.constant(2.0, shape=None, dtype=tf.float32))
    z = x*w + y**2
    return z


def another_not_so_simple_computation(w):
    x = tf.get_variable('x', initializer=tf.constant(5.0, shape=None, dtype=tf.float32))
    y = tf.get_variable('y', initializer=tf.constant(2.0, shape=None, dtype=tf.float32))
    z = x*y*w
    return z


tf.global_variables_initializer().run()

with tf.variable_scope('scopeA'):
   z1 = not_so_simple_computation(tf.constant(1.0, dtype=tf.float32))
   print(session.run(z1))

with tf.variable_scope('scopeA', reuse=True):
    z2 = another_not_so_simple_computation(z1)

session.close()

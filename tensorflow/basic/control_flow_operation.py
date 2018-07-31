import tensorflow as tf

session = tf.InteractiveSession()

######### No control flow
x = tf.Variable(tf.constant(2.0), name='x')
x_assign_op = tf.assign(x, x+5)

z = x*2

tf.global_variables_initializer().run()
print('z=', session.run(z))
print('x=', session.run(x))
session.close()

##### Control flow

session = tf.InteractiveSession()

x = tf.Variable(tf.constant(2.0), name='x')
with tf.control_dependencies([tf.assign(x, x+5)]):
    z = x*2

tf.global_variables_initializer().run()
print('z=', session.run(z))
print('x=', session.run(x))

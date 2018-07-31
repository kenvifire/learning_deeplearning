import tensorflow as tf
graph = tf.Graph()
session = tf.InteractiveSession(graph=graph)

x = tf.constant(value=[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]], dtype=tf.float32, name='x')

W = tf.Variable(tf.random_uniform(shape=[10,5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W')
b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')

h = tf.nn.sigmoid(tf.matmul(x,W) + b)

tf.global_variables_initializer().run()

h_eval = session.run(h)
print(h_eval)
session.close()
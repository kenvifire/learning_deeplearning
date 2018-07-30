import tensorflow as tf

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

tf_x = tf.Variable(tf.constant(2.0, dtype=tf.float32, name='x'))
tf_y = tf_x ** 2

minimize_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(tf_y)
tf.global_variables_initializer().run()

for i in range(0,10):
    sess.run(minimize_op)
    print(sess.run(tf_y))


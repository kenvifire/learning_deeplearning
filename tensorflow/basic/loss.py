import tensorflow as tf

sess = tf.Session()

## MSE
x = tf.constant([[2,4], [6, 8]], dtype=tf.float32)
x_hat = tf.constant([[1,2], [3,4]], dtype=tf.float32)

MSE = tf.nn.l2_loss(x-x_hat)

print(sess.run(MSE))

y = tf.constant([[1,0], [0,1]], dtype=tf.float32)
y_hat = tf.constant([[3,1], [2,5]], dtype=tf.float32)

CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y))
print(sess.run(CE))

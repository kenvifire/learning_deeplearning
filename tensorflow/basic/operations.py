import tensorflow as tf

sess = tf.Session()

## Comparasion operations
x = tf.constant([[1,2],[3,4]], dtype=tf.float32)
y = tf.constant([[4,3],[3,2]], dtype=tf.float32)

x_equal_y = tf.equal(x, y, name=None)
print(sess.run(x_equal_y))

x_less_y = tf.less(x, y, name=None)
print(sess.run(x_less_y))

x_greate_equal_y = tf.greater_equal(x, y, name=None)
print(sess.run(x_greate_equal_y))


condition = tf.constant([[True, False], [True, False]], dtype=tf.bool)
x_cond_y = tf.where(condition, x, y, name=None)
print(sess.run(x_cond_y))

## Mathmatical operations

x_add_y = tf.add(x, y)
print(sess.run(x_add_y))


x_mul_y = tf.matmul(x, y)
print(sess.run(x_mul_y))

log_x = tf.log(x)
print(sess.run(log_x))

x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)
print(sess.run(x_sum_1))

x_sum_2 = tf.reduce_sum(x, axis=[0], keepdims=True)
print(sess.run(x_sum_2))

data = tf.constant([1,2,3,4,5,6,7,8,9,10], dtype=tf.float32)
segment_data = tf.constant([0,0,0,1,1,2,2,2,2,2], dtype=tf.int32)
x_seg_sum = tf.segment_sum(data, segment_data)
print(sess.run(x_seg_sum))

## Scatter and gather operations
ref = tf.Variable(tf.constant([1,9,3,10,5], dtype=tf.float32), name='scatter_update')
tf.global_variables_initializer().run(session=sess)
indices = [1,3]
updates = tf.constant([2,4], dtype=tf.float32)
tf_scatter_update = tf.scatter_update(ref, indices, updates, use_locking=None, name=None)
print("===========")
print(sess.run(tf_scatter_update))

indices = [[1], [3]]
updates = tf.constant([[1,1,1],[2,2,2]])
shape = [4,3]
tf_scatter_nd_1 = tf.scatter_nd(indices, updates, shape, name=None)
print(sess.run(tf_scatter_nd_1))

indices = [[1,0], [3,1]]
updates = tf.constant([1,2])
shape = [4,3]
tf_scatter_nd_2 = tf.scatter_nd(indices, updates, shape, name=None)
print(sess.run(tf_scatter_nd_2))


params = tf.constant([1,2,3,4,5], dtype=tf.float32)
indices = [1,4]
tf_gather = tf.gather(params, indices, validate_indices=True, name=None)
print(sess.run(tf_gather))

params = tf.constant([[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=tf.float32)
indices = [[0], [2]]
indices = [[0,1],[2,2]]
tf_gather_nd_2 = tf.gather_nd(params, indices, name=None)
print(sess.run(tf_gather_nd_2))

### Neural network-related operations
sig_x = tf.nn.sigmoid(x, name=None)
print(sess.run(sig_x))

relu_x = tf.nn.relu(x, name=None)
print(sess.run(relu_x))

## convolution

x = tf.constant(
    [[
        [[1], [2], [3], [4]],
        [[4], [3], [2], [1]],
        [[5], [6], [7], [8]],
        [[8], [7], [6], [5]]
    ]],
    dtype=tf.float32
)

x_filter = tf.constant(
    [
        [
            [[0.5]], [[1]]
        ],
        [
            [[0.5]], [[1]]
        ]
    ],
    dtype=tf.float32
)

x_stride = [1,1,1,1]
x_padding = 'VALID'

x_conv = tf.nn.conv2d(
    input=x, filter=x_filter,
    strides=x_stride, padding=x_padding
)

print(sess.run(x_conv))

### Pooling operation
x_ksize = [1,2,2,1]
x_stride = [1,2,2,1]
x_padding = 'VALID'

x_pool = tf.nn.max_pool(
    value=x, ksize=x_ksize,
    strides=x_stride, padding=x_padding
)
print(sess.run(x_pool))


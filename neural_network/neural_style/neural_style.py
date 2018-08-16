import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf




def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, (n_H * n_W, n_C)))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, (n_H * n_W, n_C)))


    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, ((n_H * n_W), n_C)))
    a_G = tf.transpose(tf.reshape(a_G, ((n_H * n_W), n_C)))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]

        a_S = sess.run(out)
        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = J_content * alpha + J_style * beta
    return J





## pretrained vgg model, downloaded from https://www.coursera.org/learn/convolutional-neural-networks/notebook/lEthw/art-generation-with-neural-style-transfer
model = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat')
print(model)

sess = tf.InteractiveSession()
content_image = scipy.misc.imread('images/mine.jpg')
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread('images/shuimo.jpg')
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)
]


sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, alpha=10, beta=40)

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])

        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration: " + str(i) + " :")
            print("Total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image("out/generate_image.jpg", generated_image)


    save_image("out/generate_image.jpg", generated_image)

    return generated_image


model_nn(sess, generated_image)

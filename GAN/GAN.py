import os
import sys
import numpy as np
import scipy as sc
import tensorflow as tf


def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


def save_visualization(X, nh_nw, save_path='Images/sample.jpg'):
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n, x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    sc.misc.imsave(save_path, img)


def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))


def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X-mean), [0, 1, 2])
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X*g + b
    else:
        raise NotImplementedError

    return X


class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[28, 28, 1],
            dim_z=100,
            dim_y=10,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
    ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(tf.random_normal(
            [dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal(
            [dim_W1+dim_y, dim_W2*7*7], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal(
            [5, 5, dim_W3, dim_W2+dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal(
            [5, 5, dim_channel, dim_W3+dim_y], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(tf.random_normal(
            [5, 5, dim_channel+dim_y, dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal(
            [5, 5, dim_W3+dim_y, dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal(
            [dim_W2*7*7+dim_y, dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal(
            [dim_W1+dim_y, 1], stddev=0.02), name='discrim_W4')

    def build_model(self):

        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(
            tf.float32, [self.batch_size]+self.image_shape)
        h4 = self.generate(Z, Y)
        image_gen = tf.nn.sigmoid(h4)
        raw_real = self.discriminate(image_real, Y)
        p_real = tf.nn.sigmoid(raw_real)
        raw_gen = self.discriminate(image_gen, Y)
        p_gen = tf.nn.sigmoid(raw_gen)
        discrim_cost_real = bce(raw_real, tf.ones_like(raw_real))
        discrim_cost_gen = bce(raw_gen, tf.zeros_like(raw_gen))
        discrim_cost = discrim_cost_real + discrim_cost_gen

        gen_cost = bce(raw_gen, tf.ones_like(raw_gen))

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat(axis=3, values=[
                      image, yb*tf.ones([self.batch_size, 28, 28, self.dim_y])])

        h1 = lrelu(tf.nn.conv2d(X, self.discrim_W1,
                                strides=[1, 2, 2, 1], padding='SAME'))
        h1 = tf.concat(axis=3, values=[
                       h1, yb*tf.ones([self.batch_size, 14, 14, self.dim_y])])

        h2 = lrelu(batchnormalize(tf.nn.conv2d(
            h1, self.discrim_W2, strides=[1, 2, 2, 1], padding='SAME')))
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat(axis=1, values=[h2, Y])

        h3 = lrelu(batchnormalize(tf.matmul(h2, self.discrim_W3)))
        h3 = tf.concat(axis=1, values=[h3, Y])

        h4 = lrelu(batchnormalize(tf.matmul(h3, self.discrim_W4)))

        return h4

    def generate(self, Z, Y):
        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat(axis=1, values=[Z, Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat(axis=1, values=[h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size, 7, 7, self.dim_W2])
        h2 = tf.concat(axis=3, values=[
                       h2, yb*tf.ones([self.batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [self.batch_size, 14, 14, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(
            h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(batchnormalize(h3))
        h3 = tf.concat(axis=3, values=[
                       h3, yb*tf.ones([self.batch_size, 14, 14, self.dim_y])])

        output_shape_l4 = [self.batch_size, 28, 28, self.dim_channel]
        h4 = tf.nn.conv2d_transpose(
            h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        return h4

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat(axis=1, values=[Z, Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat(axis=1, values=[h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size, 7, 7, self.dim_W2])
        h2 = tf.concat(axis=3, values=[
                       h2, yb*tf.ones([batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [batch_size, 14, 14, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(
            h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(batchnormalize(h3))
        h3 = tf.concat(axis=3, values=[
                       h3, yb*tf.ones([batch_size, 14, 14, self.dim_y])])

        output_shape_l4 = [batch_size, 28, 28, self.dim_channel]
        h4 = tf.nn.conv2d_transpose(
            h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        x = tf.nn.sigmoid(h4)
        return Z, Y, x


def mnist():
    fd = open('Data/train-images-idx3-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open('Data/train-labels-idx1-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open('Data/t10k-images-idx3-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open('Data/t10k-labels-idx1-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY


def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    train_inds = np.arange(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY


n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [28, 28, 1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1

visualize_dim = 196

trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

dcgan_model = DCGAN(
    batch_size=batch_size,
    image_shape=image_shape,
    dim_z=dim_z,
    dim_W1=dim_W1,
    dim_W2=dim_W2,
    dim_W3=dim_W3,
)

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith(
    'discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = tf.train.AdamOptimizer(
    learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(
    learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(
    batch_size=visualize_dim)

tf.global_variables_initializer().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim, dim_z))
Y_np_sample = OneHot(np.random.randint(10, size=[visualize_dim]))
iterations = 0
k = 2

step = 200

for epoch in range(n_epochs):
    index = np.arange(len(trY))
    np.random.shuffle(index)
    trX = trX[index]
    trY = trY[index]

    for start, end in zip(
            range(0, len(trY), batch_size),
            range(batch_size, len(trY), batch_size)
    ):

        Xs = trX[start:end].reshape([-1, 28, 28, 1]) / 255.
        Ys = OneHot(trY[start:end])
        Zs = np.random.uniform(-1, 1,
                               size=[batch_size, dim_z]).astype(np.float32)

        if np.mod(iterations, k) != 0:
            _, gen_loss_val = sess.run(
                [train_op_gen, g_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys
                })
            discrim_loss_val, p_real_val, p_gen_val = sess.run(
                [d_cost_tf, p_real, p_gen], feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            print("=========== updating G ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)

        else:
            _, discrim_loss_val = sess.run(
                [train_op_discrim, d_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys,
                    image_tf: Xs
                })
            gen_loss_val, p_real_val, p_gen_val = sess.run(
                [g_cost_tf, p_real, p_gen], feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            print("=========== updating D ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)

        print("Average P(real)=", p_real_val.mean())
        print("Average P(gen)=", p_gen_val.mean())

        if np.mod(iterations, step) == 0:
            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: Z_np_sample,
                    Y_tf_sample: Y_np_sample
                })
            generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, (14, 14),
                               save_path='./Images/{}.jpg'.format(int(iterations/step)))

        iterations += 1
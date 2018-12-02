import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd

PATH = os.getcwd()

LOG_DIR = PATH + '/mnist-tensorboard/log-2'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

x = pd.read_csv("mnist-tensorboard/data/x_test.csv").values[:, 1:]
y = pd.read_csv("mnist-tensorboard/data/y_test.csv").values[:, 1]

images = tf.Variable(x, name='images')
with open(metadata, 'w') as metadata_file:
    for row in range(7947):
        c = y[row]
        metadata_file.write('{}\n'.format(c))

# mnist = input_data.read_data_sets(
#     PATH + "/mnist-tensorboard/data/", one_hot=True)

# images = tf.Variable(mnist.test.images, name='images')
# # def save_metadata(file):
# with open(metadata, 'w') as metadata_file:
#     for row in range(10000):
#         c = np.nonzero(mnist.test.labels[::1])[1:][0][row]
#         metadata_file.write('{}\n'.format(c))

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    print(images.shape)

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
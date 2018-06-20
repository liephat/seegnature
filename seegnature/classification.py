from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def cnn(run_label, X, Y, testX, testY):
    
    run_id = run_label + '_convnet_ern'
    time_point_number = len(X[0])
    channel_number = len(X[0][1])

    X = X.reshape([-1, time_point_number, channel_number, 1])
    testX = testX.reshape([-1, time_point_number, channel_number, 1])
    
    # Building convolutional network
    network = input_data(shape=[None, time_point_number, channel_number, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_dir='..\\out\\tflearn_logs', tensorboard_verbose=3)
    model.fit({'input': X}, {'target': Y}, n_epoch=20,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id=run_id)

def test(features, labels):

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

    iterator = dataset.make_initializable_iterator()

    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                              labels_placeholder: labels})

    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    linear_model = tf.layers.Dense(units=1)

    y_pred = linear_model(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y_pred))

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    print(sess.run(loss))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(1000):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    print(sess.run(y_pred))

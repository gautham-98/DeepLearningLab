import gin
import keras
import tensorflow as tf
import logging

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import layers, Sequential

from models.layers import vgg_block


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable()
def cnn(input_shape, base_filters, kernel_size, strides, max_pool_dim, dropout_rate):
    model = tf.keras.Sequential(name="CNN_Basic_Model_1")
    model.add(tf.keras.Input(shape=input_shape, ))
    model.add(Conv2D(filters=base_filters[0], kernel_size=kernel_size[0], strides=strides, activation="relu",
                     kernel_regularizer=regularizers.L1(l1=0.01, )))
    model.add(Conv2D(filters=base_filters[1], kernel_size=kernel_size[0], strides=strides, activation="relu",
                     kernel_regularizer=regularizers.L1(l1=0.01, )))
    model.add(MaxPool2D(pool_size=max_pool_dim))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=base_filters[2], kernel_size=kernel_size[1], strides=strides, activation="relu",
                     kernel_regularizer=regularizers.L1(l1=0.01, )))
    model.add(MaxPool2D(pool_size=max_pool_dim))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=base_filters[3], kernel_size=kernel_size[3], strides=strides, activation="relu",
                     kernel_regularizer=regularizers.L1(l1=0.01, )))
    model.add(MaxPool2D(pool_size=max_pool_dim))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=base_filters[4], kernel_size=kernel_size[3], strides=strides, activation="relu",
                     kernel_regularizer=regularizers.L1(l1=0.01, ), name='to_grad_cam'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=16, kernel_regularizer=regularizers.l2(0.001), activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=2, kernel_regularizer=regularizers.l2(0.001)))
    model.build()
    return model


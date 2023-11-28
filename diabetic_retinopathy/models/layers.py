import gin
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation

@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out

def cnn_block(input, filter, kernel_size, stride):

    conv2D = Conv2D(filters=filter,
                    kernel_size=kernel_size,
                    padding='same',
                    strides=stride,
                    kernel_regularizer=regularizers.L1(0.01)
                    #kernel_initializer =tf.keras.initializers.HeNormal()
                    )
    
    out = conv2D(input)
    out = Activation('relu')(out)
    out = BatchNormalization()(out)
  
    return out

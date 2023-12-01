import gin
import keras
import tensorflow as tf
import logging

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Add, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import InceptionResNetV2

from models.layers import vgg_block, cnn_block


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

@gin.configurable()
def cnn01(input_shape, filters, kernel_size, strides, pool_size, dropout_rate):

    model = tf.keras.Sequential(name='cnn01')
    model.add(tf.keras.Input(shape=input_shape))

    model.add(Conv2D(filters=filters[0],
              kernel_size=kernel_size[0],
              strides=strides[0],
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=filters[1],
              kernel_size=kernel_size[1],
              strides=strides[1],
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    #model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=filters[2],
              kernel_size=kernel_size[2],
              strides=strides[2],
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    #model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=filters[3],
              kernel_size=kernel_size[3],
              strides=strides[3],
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    #model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=filters[4],
              kernel_size=kernel_size[4],
              strides=strides[4],
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())
    # model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(Conv2D(filters=filters[5],
              kernel_size=kernel_size[5],
              strides=strides[5],
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(Conv2D(filters=filters[6],
              kernel_size=kernel_size[6],
              strides=strides[6],
              dilation_rate = 1,
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal(),
              name='to_grad_cam'
              ))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(Conv2D(filters=filters[7],
              kernel_size=kernel_size[7],
              strides=strides[7],
              dilation_rate = 1,
              activation='relu',
              kernel_regularizer=regularizers.L1(l1=0.01),
              kernel_initializer =tf.keras.initializers.HeNormal()
              ))
    #model.add(MaxPool2D(pool_size=pool_size))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=32, kernel_regularizer=regularizers.l1(0.001), activation = 'relu', kernel_initializer =tf.keras.initializers.HeNormal()))
    #model.add(tf.keras.layers.Dropout(dropout_rate))

    # model.add(tf.keras.layers.Dense(
    #     units=16, kernel_regularizer=regularizers.l1(0.001), kernel_initializer =tf.keras.initializers.HeNormal()))
    # model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=8, kernel_regularizer=regularizers.l1(0.001), activation = 'relu', kernel_initializer =tf.keras.initializers.HeNormal()))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    # model.add(tf.keras.layers.Dense(
    #     units=4, kernel_regularizer=regularizers.l1(0.001), kernel_initializer =tf.keras.initializers.HeNormal()))
    # model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=2, kernel_regularizer=regularizers.l1(0.001), kernel_initializer =tf.keras.initializers.HeNormal()))

    model.build()

    logging.info(f"cnn01 input shape:  {model.input_shape}")
    logging.info(f"cnn01 output shape: {model.output_shape}")

    return model

@gin.configurable()
def res_cnn(input_shape, filters, kernel_size, strides, pool_size, dropout_rate, maxpool_blocks,  dropout_blocks=False, res_blocks=False):
    
    inputs = Input(shape=input_shape)


    # convolutional layers
    out = []
    # first convolution layer
    out_block1 = cnn_block(inputs, 
                        filters[0], 
                        kernel_size, 
                        strides[0]
                        )
    out.append(out_block1)

    for block in range(1,len(filters)):
        #Conv2d
        out_block = cnn_block(out[-1], 
                        filters[block], 
                        kernel_size, 
                        strides[block]
                        )
        out.append(out_block)

        #Skip connection
        if res_blocks:
            for skip_connection in res_blocks:
                if block == skip_connection[1]:
                    out_to_add = out[skip_connection[0]]
                    # reshape for equal number of feature maps
                    if not (out[-1].shape[-1] == out_to_add.shape[-1]):
                        out_to_add = Conv2D(filters=out[-1].shape[-1], 
                                            kernel_size=(1,1), 
                                            kernel_regularizer=regularizers.L1(0.01)
                                            )(out_to_add)
                        out_to_add = BatchNormalization()(out_to_add)
                    out[-1] = Add()([out_to_add, out[-1]])
        
        #dropout and maxpooling
        if block in maxpool_blocks:
            out[-1] = MaxPool2D(pool_size)(out[-1])
        if dropout_blocks: 
            if block in dropout_blocks:
                out[-1] = Dropout(dropout_rate)(out[-1])

    # dense layers
    out_dense = GlobalAveragePooling2D()(out[-1])
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=32, kernel_regularizer=regularizers.l2(0.01), activation='relu')(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=16, kernel_regularizer=regularizers.l2(0.01), activation='relu')(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=4, kernel_regularizer=regularizers.l2(0.01))(out_dense)

    outputs = Dense(units=2)(out_dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="res_cnn")
    model.build(input_shape=input_shape)

    logging.info(f"res_cnn input shape:  {model.input_shape}")
    logging.info(f"res_cnn output shape: {model.output_shape}")

    return model

@gin.configurable
def transfer_model(input_shape, filters, dense_units, dropout_rate):

    """returns both the whole model and the base_model
      further steps for making the layers trainable are
      done by TransferTrainer"""

    inputs = Input(shape=input_shape)

    base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                         weights="imagenet", 
                                                         input_shape=input_shape, 
                                                         pooling=None)
    
    out = base_model(inputs)
    out = Conv2D(filters=filters, kernel_size=3, strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.01))(out)
    out = BatchNormalization()(out) 

    out_dense = GlobalAveragePooling2D()(out)
    out_dense = Dense(units=int(dense_units/2), kernel_regularizer=regularizers.l2(0.01), activation='relu')(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=int(dense_units/4), kernel_regularizer=regularizers.l2(0.01))(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)

    outputs = Dense(units=2)(out_dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="transfer_model")

    logging.info(f"transfer_model input shape:  {model.input_shape}")
    logging.info(f"transfer_model output shape: {model.output_shape}")

    return model, base_model


    







import gin
import keras
import tensorflow as tf
import logging

from tensorflow.keras import regularizers
from tensorflow.keras import layers

def one_lstm_layer(lstm_cells, dropout_rate):
    return layers.LSTM(lstm_cells, dropout=dropout_rate,return_sequences=True)

@gin.configurable
def model1_LSTM(window_length,num_lstm,dense_units,lstm_cells,n_classes,dropout_rate=0.3):
    model = keras.Sequential([keras.Input(shape=(window_length,6))])
    for e in range(num_lstm):
        layer = one_lstm_layer(lstm_cells, dropout_rate=dropout_rate)
        model.add(layer)
        model.add(layers.BatchNormalization())
    model.add(layers.LSTM(lstm_cells, dropout=dropout_rate))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(dense_units , activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_classes, activation="linear"))
    model.build()
    print(model.input_shape)
    print(model.output_shape)


    return model



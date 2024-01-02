import gin
import keras
import tensorflow as tf
import logging

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Add, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import layers, Sequential


@gin.configurable
def model1_LSTM(window_length, dropout=0.3):
    
    model = keras.Sequential([keras.Input(shape=(window_length,6))])
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(dropout))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64 , activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(12, activation="linear"))
    model.build()

    return model



"""
Collection of various sequential layers in keras. Many of these models
are directly copied from keras examples
"""
import keras
from keras.models import Sequential
from keras.layers import (Dense,
                          Activation,
                          Dropout,
                          Conv2D,
                          Flatten,
                          MaxPooling2D,
                          ConvLSTM2D,
                          Conv3D,
                          LSTM,
                          BatchNormalization)
from keras.constraints import MinMaxNorm, MaxNorm
from keras.initializers import glorot_normal, glorot_uniform
from keras.regularizers import l1_l2


def simple_dense():
    """Creates a simple sequential model, with 5 dense layers"""
    model = Sequential()
    model.add(Dense(units=32,
                    input_shape=(32,),
                    use_bias=True,
                    bias_constraint=MinMaxNorm(min_value=-1,
                                               max_value=1,
                                               rate=1.0,
                                               axis=0),
                    bias_initializer=glorot_normal(seed=32),
                    kernel_constraint=MaxNorm(max_value=1.5),
                    kernel_initializer=glorot_uniform(seed=45)))
    model.add(Activation('relu'))
    model.add(Dense(units=32,
                    activation='tanh',
                    use_bias=False,
                    activity_regularizer=l1_l2(l1=0.05, l2=0.05),
                    kernel_constraint=MaxNorm(max_value=1.5),
                    kernel_initializer=glorot_uniform(seed=45)))

    model.add(Dense(units=10,
                    activation='softmax',
                    use_bias=False,
                    activity_regularizer=l1_l2(l1=0.05, l2=0.05),
                    kernel_constraint=MaxNorm(max_value=1.5),
                    kernel_initializer=glorot_uniform(seed=45)))
    return model


class SequentialSubClass(Sequential):
    """A Model Class in keras that subclasses the sequential Model"""
    def __init__(self):
        super(SequentialSubClass, self).__init__()

    def load_model(self, num_layers=10):
        self.add(Dense(units=32,
                       input_shape=(32,),
                       use_bias=True,
                       bias_constraint=MinMaxNorm(min_value=-1,
                                                  max_value=1,
                                                  rate=1.0,
                                                  axis=0),
                       bias_initializer=glorot_normal(seed=32),
                       kernel_constraint=MaxNorm(max_value=1.5),
                       kernel_initializer=glorot_uniform(seed=45)))
        self.add(Dense(units=32,
                       use_bias=True,
                       activation='tanh',
                       bias_constraint=MinMaxNorm(min_value=-1,
                                                  max_value=1,
                                                  rate=1.0,
                                                  axis=0),
                       bias_initializer=glorot_normal(seed=32),
                       kernel_constraint=MaxNorm(max_value=1.5),
                       kernel_initializer=glorot_uniform(seed=45)))
        self.add(Dropout(rate=0.5))

        self.add(Dense(units=10,
                       use_bias=True,
                       activation='softmax',
                       bias_constraint=MinMaxNorm(min_value=-1,
                                                  max_value=1,
                                                  rate=1.0,
                                                  axis=0),
                       bias_initializer=glorot_normal(seed=32),
                       kernel_constraint=MaxNorm(max_value=1.5),
                       kernel_initializer=glorot_uniform(seed=45)))


def seq_conv_mnist():
    """A sequential convolutional model with pooling layers.
    This model is in keras-example from mnist convolutional classification
    here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3,), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def seq_conv_cifar():
    """A Sequential Model written in keras to classify the cifar10 dataset
    This is available at
    https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    """
    model = Sequential()
    model.add(Conv2D(32,
                     (3, 3),
                     padding='same',
                     input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def conv_lstm():
    """This model demonstrates the use of a convolutional LSTM network.
    This network is used to predict the next frame of an artificially
    generated movie which contains moving squares.
    Copied from:
        https://github.com/keras-team/keras/blob/master/examples/conv_lstm.py
    """
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(None, 40, 40, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    return seq


def seq_lstm():
    """A sequential character level LSTM Language Model
    Copied From:
    https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(500, 30)))
    model.add(Dense(35, activation='softmax'))

    return model

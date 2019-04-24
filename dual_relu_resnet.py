from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    BatchNormalization
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers import (Add,Lambda,Concatenate)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from resnet import (
    _handle_dim_ordering,
    _shortcut
)


def _relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        relu = Activation("relu")(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(relu)

    return f

def _invRelu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        invert = Lambda(lambda x: x*(-1))(input)
        relu = Activation("relu")(invert)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(relu)

    return f

def dual_relu_residual(option,**conv_params):
    #filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    concatenate_mode = option["concatenate"]
    if concatenate_mode == "half_concanate":
        filters = int(conv_params["filters"]/2)
    else:
        filters = conv_params["filters"]

    def f(input):
        AXIS = _handle_dim_ordering()
        BN = BatchNormalization(axis=AXIS[2])(input)
        positive_conv = _relu_conv(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(BN)
        negative_conv = _invRelu_conv(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(BN)
        if concatenate_mode == "half_concanate" or concatenate_mode == "full_concanate":
            return Concatenate(axis=3)([positive_conv,negative_conv])
        elif concatenate_mode == "sum":
            return Add()([positive_conv,negative_conv])
    return f


def dual_relu_basic_block(filters, option,init_strides=(1, 1), is_first_block_of_first_layer=False):

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = dual_relu_residual(filters=filters, option=option,kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = dual_relu_residual(filters=filters, option=option,kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def dual_relu_bottleneck(filters, option,init_strides=(1, 1), is_first_block_of_first_layer=False):

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = dual_relu_residual(filters=filters, option=option,kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = dual_relu_residual(filters=filters, option=option,kernel_size=(3, 3))(conv_1_1)
        residual = dual_relu_residual(filters=filters * 4, option=option, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f
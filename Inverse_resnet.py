# -*- coding: utf-8 -*-
'''
    コピペ元
    http://pynote.hatenablog.com/entry/keras-resnet-implementation
'''

from functools import reduce

from keras import backend as K
from keras.layers import (Activation, Add, Multiply,GlobalAveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Flatten, Input,Dropout,
                          MaxPooling2D,Lambda)
from keras.models import Model
from keras.regularizers import l2
import numpy as np

from base import(compose,shortcut,ResNetConv2D)

def inv_relu_conv(*args,**kwargs):
    return compose(
        Lambda(lambda x: x*(-1)),
        Activation('relu'),
        ResNetConv2D(*args, **kwargs)
    )

def relu_conv(*args, **kwargs):
    return compose(
        Activation('relu'),
        ResNetConv2D(*args, **kwargs)
    )

def inv_basic_block(filters, first_strides, is_first_block_of_first_layer):

    def f(x):
        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            BN = BatchNormalization()(x)
            positive_conv1 = relu_conv(filters=filters, kernel_size=(3, 3),
                                strides=first_strides)(BN)
            negative_conv1 = inv_relu_conv(filters=filters, kernel_size=(3, 3),
                                strides=first_strides)(BN)
            conv1 = Add()([positive_conv1,negative_conv1])
        
        BN2 = BatchNormalization()(conv1)
        positive_conv2 = relu_conv(filters=filters, kernel_size=(3, 3))(BN2)
        negative_conv2 = inv_relu_conv(filters=filters, kernel_size=(3, 3))(BN2)
        conv2 = Add()([positive_conv2,negative_conv2])

        return shortcut(x,conv2)
    
    return f

def inv_bottleneck(filters, first_strides, is_first_block_of_first_layer):
    
    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            BN = BatchNormalization()(x)
            positive_conv1 = relu_conv(filters=filters, kernel_size=(1, 1),
                                strides=first_strides)(BN)
            negative_conv1 = inv_relu_conv(filters=filters, kernel_size=(1, 1),
                                strides=first_strides)(BN)
            conv1 = Add()([positive_conv1,negative_conv1])

        BN2 = BatchNormalization()(conv1)
        positive_conv2 = relu_conv(filters=filters, kernel_size=(3, 3))(BN2)
        negative_conv2 = inv_relu_conv(filters=filters, kernel_size=(3, 3))(BN2)
        conv2 = Add()([positive_conv2,negative_conv2])

        BN3 = BatchNormalization()(conv2)
        positive_conv3 = relu_conv(filters=filters, kernel_size=(3, 3))(BN3)
        negative_conv3 = inv_relu_conv(filters=filters, kernel_size=(3, 3))(BN3)
        conv3 = Add()([positive_conv3,negative_conv3])

        #conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        #conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)

        return shortcut(x, conv3)

    return f

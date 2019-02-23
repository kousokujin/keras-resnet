from resnet import (
    _handle_dim_ordering,
    _get_block,_bn_relu,
    _conv_bn_relu,MaxPooling2D,
    _residual_block,
    AveragePooling2D,
    basic_block,
    bottleneck
)
from dual_relu_resnet import(
    dual_relu_basic_block,
    dual_relu_bottleneck,
    dual_relu_residual
)

from Invert_ReLu_resnet import _invert_bn_relu
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras import backend as K
from keras.models import Model

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions,option = False):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        AXIS = _handle_dim_ordering()
        ROW_AXIS = AXIS[0]
        COL_AXIS = AXIS[1]

        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        #if (block_fn == basic_block) or (block_fn == bottleneck):
        #    conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        #else:
        #    conv1 = dual_relu_residual(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0),option=option)(block)
            filters *= 2

        # Last activation
        if (block_fn == basic_block) or (block_fn == bottleneck):
            if option:
                block = _invert_bn_relu(block)
            else:
                block = _bn_relu(block)
        else:
            block = _bn_relu(block)

        # Classifier block
        #ROW_AXIS = _handle_dim_ordering().ROW_AXIS
        #COL_AXIS = _handle_dim_ordering().COL_AXIS

        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
    
    @staticmethod
    def build_dualresnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_dualresnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_dualresnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_dualresnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_dualresnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 8, 36, 3])

    #------------------------
    # option is true models
    #------------------------

    @staticmethod
    def build_invert_relu_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2], option=True)

    @staticmethod
    def build_invert_relu_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3], option=True)

    @staticmethod
    def build_invert_relu_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3], option=True)

    @staticmethod
    def build_invert_relu_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3], option=True)

    @staticmethod
    def build_invert_relu_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3], option=True)
    
    @staticmethod
    def build_concatenate_dualresnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [2, 2, 2, 2], option=True)

    @staticmethod
    def build_concatenate_dualresnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [3, 4, 6, 3], option=True)

    @staticmethod
    def build_concatenate_dualresnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 6, 3], option=True)

    @staticmethod
    def build_concatenate_dualresnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 23, 3], option=True)

    @staticmethod
    def build_concatenate_relu_dualresnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 8, 36, 3], option=True)
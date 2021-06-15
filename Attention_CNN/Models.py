from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Flatten


class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                  strides=stride, padding='SAME', use_bias=False, name='Conv2d')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm')
        self.activation = layers.ReLU(max_value=6.0)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedResidual(layers.Layer):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer_list = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))

        layer_list.extend([
            # 3x3 depthwise conv
            layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                   use_bias=False, name='depthwise'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='depthwise/BatchNorm'),
            layers.ReLU(max_value=6.0),
            # 1x1 pointwise conv(linear)
            layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                          padding='SAME', use_bias=False, name='project'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='project/BatchNorm')
        ])
        self.main_branch = Sequential(layer_list, name='expanded_conv')

    def call(self, inputs, training=False, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs, training=training)
        else:
            return self.main_branch(inputs, training=training)


def MobileNetV2(Chans=64,
                Samples=128,
                num_classes=3,
                alpha=1.0,
                round_nearest=8,
                include_top=True, dropoutRate=0.5):
    block = InvertedResidual
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(10, (1, 5), padding="same",
                    input_shape=(1, Chans, Samples), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    # block1 = BatchNormalization(axis=1)(block1)
    block1 = Conv2D(10, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1) \
        (block1)  # axis=1:channel_first,data_format="batch_shape + (rows, cols, channels)"
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = block(in_channel=10, out_channel=16, stride=1, expand_ratio=4)(block1)

    # input_channel = _make_divisible(32 * alpha, round_nearest)
    # last_channel = _make_divisible(1280 * alpha, round_nearest)

    # input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    # conv1
    # x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
    # building inverted residual residual blockes
    # for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
    #     output_channel = _make_divisible(c * alpha, round_nearest)
    #     for i in range(n):
    #         stride = s if i == 0 else 1
    #         x = block(x.shape[-1],
    #                   output_channel,
    #                   stride,
    #                   expand_ratio=t)(x)
    # # building last several layers
    x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)

    if include_top is True:
        # building classifier
        x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes, name='Logits')(x)
    else:
        output = x

    model = Model(inputs=input_image, outputs=output)
    return model

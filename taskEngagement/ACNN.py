from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras import backend as K


def DeepConvNet_V1(nb_classes, Chans=64, Samples=128,
                   dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """
    # max_norm:最大范数权值约束
    # start the model
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

    block2 = Conv2D(20, (1, 5), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(40, (1, 5), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    # block4 = Conv2D(80, (1, 5), use_bias=False,
    #                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    # block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    # block4 = Activation('elu')(block4)
    # block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    # block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block3)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

# print("---------测试模型---------")
# model = DeepConvNet_V1(3)
# model.summary()
# # for layer in model.layers:
#     print(layer.name)
# # layers_name=[]

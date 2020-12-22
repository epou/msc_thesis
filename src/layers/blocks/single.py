from tensorflow.keras import layers

from src.layers.baselayer import BaseLayer


class ConvBlock(BaseLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), dropout_rate=None, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate

    def call(self, inputs):

        conv = inputs

        for _ in range(2):
            conv = layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding='same'
            )(conv)
            conv = layers.BatchNormalization()(conv)
            conv = layers.LeakyReLU()(conv)

        if self.dropout_rate:
            conv = layers.Dropout(rate=self.dropout_rate)(conv)
        return conv


class ResidualBlock(ConvBlock):

    def __init__(self, force_shortcut_conv=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.force_shortcut_conv = force_shortcut_conv

    def call(self, inputs):
        shortcut = inputs

        # down-sampling is performed with a stride of 2
        conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same'
        )(inputs)
        conv = layers.BatchNormalization()(conv)
        conv = layers.LeakyReLU()(conv)

        conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same'
        )(conv)
        conv = layers.BatchNormalization()(conv)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if self.strides != (1, 1) or self.force_shortcut_conv:
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                strides=self.strides,
                padding='same'
            )(inputs)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, conv])
        y = layers.LeakyReLU()(y)

        return y

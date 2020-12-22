from tensorflow.keras import layers

from src.layers.blocks.group import ConvBlockGroup, ResNetBlockGroup
from .base import Encoder


class SimpleEncoder(Encoder):
    def __init__(self, depth, feature_maps=4, **kwargs):
        super(SimpleEncoder, self).__init__(
            block_group_tupl=[(1, ConvBlockGroup) for _ in range(depth)],
            feature_maps=feature_maps,
            **kwargs
        )

    def call(self, inputs):
        y = inputs

        for i in range(self.depth):
            nb_iterations = self.block_group_tupl[i][0]
            block_group = self.block_group_tupl[i][1]
            block = block_group(
                nb_iterations=nb_iterations,
                filters=self.feature_maps * 2 ** i,
                kernel_size=(3, 3)
            )
            self.blocks[i] = block
            y = block(y)

            if i != len(self.block_group_tupl) - 1 and self.max_pool:
                y = layers.MaxPooling2D((2, 2), padding='same')(y)

        return y


class ResnetEncoder(Encoder):
    def __init__(self, nb_list, feature_maps=4, **kwargs):
        super(ResnetEncoder, self).__init__(
            block_group_tupl=[(num, ResNetBlockGroup) for num in nb_list],
            feature_maps=feature_maps,
            max_pool=False,
            **kwargs
        )
        self.num_downsamplings += 1
        self.decoder_offset = 1

    def call(self, inputs):
        y = ConvBlockGroup(
            nb_iterations=1,
            filters=self.feature_maps,
            kernel_size=(3, 3)
        )(inputs)
        y = layers.MaxPooling2D((2, 2), padding='same')(y)

        for i in range(self.depth):
            nb_iterations = self.block_group_tupl[i][0]
            block_group = self.block_group_tupl[i][1]
            block = block_group(
                first_stage=i == 0,
                nb_iterations=nb_iterations,
                filters=self.feature_maps * 2 ** (i + 1),
                kernel_size=(3, 3)
            )
            self.blocks[i] = block
            y = block(y)

        return y

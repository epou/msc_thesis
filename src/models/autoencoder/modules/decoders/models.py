from tensorflow.keras import layers

from src.layers.blocks.group import ConvBlockGroup
from src.models.autoencoder.modules.base import Coder

class Decoder(Coder):

    def __init__(self, depth, skipped_blocks={}, skip_function=None, offset=0, **kwargs):

        block_group_tupl = [(1, ConvBlockGroup) for _ in range(depth)]

        super(Decoder, self).__init__(block_group_tupl=block_group_tupl, **kwargs)
        self.skip_function = skip_function
        self.skipped_blocks = skipped_blocks
        self.offset = offset

    def call(self, inputs):
        previous_layer = inputs

        for i, i_rev in enumerate(reversed(range(self.depth - 1))):
            previous_layer = layers.UpSampling2D((2, 2))(previous_layer)
            previous_layer = layers.Conv2D(filters=self.feature_maps * 2 ** i_rev,
                                           kernel_size=(2, 2),
                                           activation='relu',
                                           padding='same',
                                           kernel_initializer='he_normal')(previous_layer)
            if self.skip_function:
                index = i_rev - self.offset
                if self.skipped_blocks.get(index):
                    previous_layer = self.skip_function(
                        [previous_layer, self.skipped_blocks.get(index).output]
                    )

            nb_iterations = self.block_group_tupl[i][0]
            block_group = self.block_group_tupl[i][1]

            block = block_group(
                nb_iterations=nb_iterations,
                filters=self.feature_maps * 2 ** i_rev,
                kernel_size=(3, 3)
            )
            self.blocks[i_rev] = block
            previous_layer = block(previous_layer)

        return layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(previous_layer)

from tensorflow import keras

from .modules.decoders.models import Decoder


class AutoEncoder(object):
    def __init__(self, encoder, name, skip_long_connections_function=None, **kwargs):
        self.name = name
        self.skip_long_connections_function = skip_long_connections_function

        self.encoder = encoder
        self.decoder = Decoder(
            depth=encoder.num_downsamplings,
            skipped_blocks=encoder.blocks,
            skip_function=skip_long_connections_function if skip_long_connections_function else None,
            offset=encoder.decoder_offset,
            feature_maps=encoder.feature_maps
        )

    def __call__(self, input_shape):
        inputs = keras.Input(shape=input_shape)

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return keras.Model(inputs, decoded, name=self.name)

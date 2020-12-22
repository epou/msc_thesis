from src.models.autoencoder.modules.base import Coder


class Encoder(Coder):

    def __init__(self, max_pool=True, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.max_pool = max_pool
        self.num_downsamplings = self.depth
        self.decoder_offset = 0

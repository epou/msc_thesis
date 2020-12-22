from src.layers.baselayer import BaseLayer


class Coder(BaseLayer):
    def __init__(self, block_group_tupl, feature_maps=4, **kwargs):
        super(Coder, self).__init__(**kwargs)

        self.block_group_tupl = block_group_tupl
        self.depth = len(block_group_tupl)

        self.feature_maps = feature_maps
        self.blocks = {}
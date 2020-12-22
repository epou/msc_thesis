from .single import ConvBlock, ResidualBlock


class ConvBlockGroup(ConvBlock):

    def __init__(self, nb_iterations=1, block_cls=ConvBlock, **kwargs):
        super(ConvBlockGroup, self).__init__(**kwargs)

        self.block_cls = block_cls
        self.nb_iterations = nb_iterations
        self.kwargs = kwargs

    def call(self, inputs):
        y = inputs
        for _ in range(self.nb_iterations):
            y = self.block_cls(**self.kwargs)(y)
        return y


class ResNetBlockGroup(ConvBlockGroup):
    def __init__(self, first_stage=False, **kwargs):

        kwargs = kwargs
        kwargs["strides"] = (1, 1)

        super(ResNetBlockGroup, self).__init__(block_cls=ResidualBlock, **kwargs)

        self.first_stage = first_stage

    def call(self, inputs):

        kwargs = self.kwargs

        y = inputs
        first = True
        for _ in range(self.nb_iterations):
            if first:
                first = False
                first_kwargs = kwargs.copy()
                first_kwargs["strides"] = (2, 2) if not self.first_stage else (1, 1)
                first_kwargs["force_shortcut_conv"] = self.first_stage
                y = self.block_cls(**first_kwargs)(y)
            else:
                y = self.block_cls(**kwargs)(y)

        return y

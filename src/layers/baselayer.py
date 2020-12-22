class BaseLayer(object):
    def __init__(self, *args, **kwargs):
        self.input = None
        self.output = None

    def __call__(self, inputs, *args, **kwargs):
        self.input = inputs

        output = self.call(inputs)
        self.output = output
        return output

    def call(self, *args, **kwargs):
        raise NotImplementedError

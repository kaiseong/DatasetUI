class Transform:
    def __call__(self, *inputs):
        return self.forward(*inputs)
    def forward(self, *inputs):
        return inputs[0] if len(inputs) == 1 else inputs
    def _call_kernel(self, function, *args, **kwargs):
        return function(*args, **kwargs)

class Identity(Transform):
    pass

class ColorJitter(Transform):
    def __init__(self, **kwargs):
        pass

class RandomAffine(Transform):
    def __init__(self, **kwargs):
        pass

from . import functional

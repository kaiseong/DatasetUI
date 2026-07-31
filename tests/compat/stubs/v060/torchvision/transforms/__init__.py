class ToTensor:
    def __call__(self, value):
        import numpy as np
        from torch import Tensor
        array = np.asarray(value)
        if array.ndim == 3:
            array = array.transpose(2, 0, 1)
        return Tensor(array.astype(np.float32) / 255.0)

from . import v2

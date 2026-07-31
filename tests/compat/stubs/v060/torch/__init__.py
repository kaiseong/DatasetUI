"""Metadata/full-loader torch stub for official fixture compatibility checks."""
import numpy as _np

class Tensor:
    def __init__(self, value=None):
        self._value = _np.asarray(value)
    def numpy(self):
        return self._value
    def item(self):
        return self._value.item()
    def __len__(self):
        return len(self._value)
    def __getitem__(self, key):
        value = self._value[key]
        return Tensor(value) if isinstance(value, _np.ndarray) else value
    @property
    def shape(self):
        return self._value.shape

class Generator:
    pass

class dtype:
    pass

class device:
    def __init__(self, value="cpu"):
        self.type = str(value)
    def __str__(self):
        return self.type

class _Unavailable:
    @staticmethod
    def is_available():
        return False

class _Backends:
    mps = _Unavailable()

cuda = _Unavailable()
backends = _Backends()
float16 = "float16"
bfloat16 = "bfloat16"
float32 = "float32"
float64 = "float64"
uint8 = "uint8"
int32 = "int32"
int64 = "int64"
long = "int64"
bool = "bool"
FloatTensor = Tensor
LongTensor = Tensor


def is_tensor(value):
    return isinstance(value, Tensor)


def tensor(value, *args, **kwargs):
    return Tensor(value)


def LongTensor(value):
    return Tensor(_np.asarray(value, dtype=_np.int64))


def stack(values, *args, **kwargs):
    arrays = [value.numpy() if isinstance(value, Tensor) else _np.asarray(value) for value in values]
    return Tensor(_np.stack(arrays, axis=kwargs.get("dim", 0)))

from . import nn

from . import optim

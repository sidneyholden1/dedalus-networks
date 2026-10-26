"""Compact canonical indices and explicitly local graph tables."""
import hashlib
from types import SimpleNamespace

import numpy as np


class CanonicalIndices:
    """Read-only identity indexing without storing a global integer vector."""
    dtype = np.dtype(np.int64)
    ndim = 1
    flags = SimpleNamespace(writeable=False)

    def __init__(self, size):
        object.__setattr__(self, 'size', int(size))
        object.__setattr__(self, 'shape', (int(size),))

    def __setattr__(self, name, value):
        raise AttributeError('Canonical indices are immutable')

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(range(self.size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return np.arange(*index.indices(self.size), dtype=np.int64)
        indices = np.asarray(index)
        if indices.dtype.kind == 'b':
            if indices.shape != self.shape:
                raise IndexError('Boolean index has the wrong shape')
            return np.flatnonzero(indices)
        if indices.dtype.kind not in 'iu':
            raise IndexError('Canonical indices require integer selection')
        result = np.where(indices < 0, indices + self.size, indices).astype(np.int64)
        if np.any(result < 0) or np.any(result >= self.size):
            raise IndexError('Canonical index out of bounds')
        return result

    def __array__(self, dtype=None, copy=None):
        return np.arange(self.size, dtype=dtype or self.dtype)

    def copy(self):
        return np.asarray(self)

    def tolist(self):
        return list(range(self.size))

    def __reduce__(self):
        return (CanonicalIndices, (self.size,))


def take_elements(values, indices):
    if isinstance(values, CanonicalIndices):
        return values[indices]
    return np.asarray(values, dtype=np.int64)[indices]


def elements_signature(values):
    if isinstance(values, CanonicalIndices):
        return ('canonical-indices-v1', values.size)
    array = np.asarray(values, dtype=np.int64)
    if array.ndim == 1 and np.array_equal(array, np.arange(array.size)):
        return ('canonical-indices-v1', array.size)
    array = np.ascontiguousarray(array)
    return (array.shape, array.dtype.str, hashlib.blake2b(memoryview(array).cast('B'), digest_size=16).digest())


class LocalGraphTable:
    """A global-indexed table storing only explicitly supplied entries.

    Missing nonlocal entries are errors, never implicit zeros. This makes
    accidental whole-graph operations visible in a distributed input path.
    """
    def __init__(self, size, indices, values, global_sum=None):
        self.indices = np.array(indices, dtype=np.int64, copy=True)
        self.values = np.array(values, copy=True)
        if self.indices.ndim != 1 or self.values.shape[0] != self.indices.size:
            raise ValueError('Invalid local graph table')
        if np.any(self.indices[1:] <= self.indices[:-1]):
            raise ValueError('Local graph indices must be sorted and unique')
        self.indices.flags.writeable = self.values.flags.writeable = False
        self.shape = (int(size),) + self.values.shape[1:]
        self.size = int(np.prod(self.shape))
        self.ndim = len(self.shape)
        self.dtype = self.values.dtype
        self.global_sum = global_sum

    def __getitem__(self, index):
        trailing = ()
        if isinstance(index, tuple):
            index, trailing = index[0], index[1:]
        if isinstance(index, slice):
            index = np.arange(*index.indices(self.shape[0]))
        labels = np.asarray(index, dtype=np.int64)
        positions = np.searchsorted(self.indices, labels)
        if np.any(positions >= self.indices.size) or np.any(self.indices[positions] != labels):
            raise IndexError('Requested graph entries are not stored on this rank')
        # Preserve NumPy paired advanced indexing of endpoint rows and sides.
        return self.values[(positions,) + trailing]

    def __array__(self, dtype=None, copy=None):
        raise TypeError('A local graph table cannot be materialized as a replicated global array')

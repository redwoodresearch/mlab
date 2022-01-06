from typing import Callable, Optional, Tuple

import numpy as np

from . import operators
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
    TensorData,
)


def tensor_map(fn: Callable[[float], float]):
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (1d array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (1d array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(
        out: np.ndarray,
        out_shape,
        out_strides,
        in_storage: np.ndarray,
        in_shape,
        in_strides,
    ):
        out_size = operators.prod_int(out_shape)
        out_index = np.array(out_shape)
        in_index = np.array(in_shape)
        for i in range(out_size):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)

            out[out_pos] = fn(in_storage[in_pos])

    return _map


def map(fn: Callable[[float], float]):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a: TensorData, out: Optional[TensorData] = None) -> TensorData:
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn: Callable[[float, float], float]):
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out: np.ndarray,
        out_shape,
        out_strides,
        a_storage: np.ndarray,
        a_shape,
        a_strides,
        b_storage: np.ndarray,
        b_shape,
        b_strides,
    ):
        out_size = operators.prod_int(out_shape)
        out_index = np.array(out_shape)
        a_index = np.array(a_shape)
        b_index = np.array(b_shape)
        for i in range(out_size):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)

            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def zip(fn: Callable[[float, float], float]):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a: TensorData, b: TensorData) -> TensorData:
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn: Callable[[float, float], float]):
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out: np.ndarray,
        out_shape,
        out_strides,
        a_storage: np.ndarray,
        a_shape,
        a_strides,
        reduce_dim: int,
    ):
        a_size = operators.prod_int(a_shape)
        a_index = np.array(a_shape)
        out_index = np.array(out_shape)
        for i in range(a_size):
            to_index(i, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)

            out_index = a_index.copy()
            out_index[reduce_dim] = 0
            out_pos = index_to_position(out_index, out_strides)

            out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce


def reduce(fn: Callable[[float, float], float], start: float = 0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`TensorData` : new tensor
    """
    f = tensor_reduce(fn)

    def ret(a: TensorData, dim: int) -> TensorData:
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce

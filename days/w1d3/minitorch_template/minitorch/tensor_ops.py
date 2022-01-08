import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def tensor_map(fn):
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
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):       
        #iterate over out_storage
        # for each position, find the indices this corresponds to in in_storage
        # find the position this corresponds to in in_storage
        for i in range(len(out)):
            in_pos, out_pos = map_positions(i, out_shape, out_strides, in_shape, in_strides)
            
            out[out_pos]=fn(in_storage[in_pos])
            

    return _map

def map_positions(i, out_shape, out_strides, in_shape, in_strides):
        out_idx = [0] * len(out_shape)

        to_index(i, out_shape, out_idx)
        out_pos = index_to_position(out_idx, out_strides)


        small_out_index = [0] * len(in_shape)

        broadcast_index(out_idx, out_shape, in_shape, small_out_index)

        # out_idx is index of out value within out_shape 
        # want in_idx which is corresponding index in in_shape

        in_pos = index_to_position(small_out_index, in_strides)
        
        return in_pos, out_pos


def map(fn):
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

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
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
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        for i in range(len(out)):
            
            a_pos, out_pos = map_positions(i, out_shape, out_strides, a_shape, a_strides)
            b_pos, _ = map_positions(i, out_shape, out_strides, b_shape, b_strides)
            
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def zip(fn):
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

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
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

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        for i in range(len(out)):
            # out_pos,a_pos  = map_positions(i, a_shape, a_strides, out_shape, out_strides) 
            out_idx = [0] * len(out_shape)
            to_index(i, out_shape, out_idx)
            out_pos = index_to_position(out_idx, out_strides)
            
            a_idx = out_idx.copy()
            a_idx[reduce_dim] = 0
            a_pos = index_to_position(out_idx, a_strides)
            accum = a_storage[a_pos]
            for j in range(1, a_shape[reduce_dim]):
                a_idx[reduce_dim] = j
                a_pos = index_to_position(a_idx, a_strides)
                accum=fn(accum, a_storage[a_pos])
            out[out_pos]=accum

    return _reduce


def reduce(fn, start=0.0):
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

    def ret(a, dim):
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

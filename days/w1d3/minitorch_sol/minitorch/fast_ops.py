import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
)
from numba import njit, prange


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.
    Optimizations:
        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing
    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.
    Returns:
        None : Fills in `out`
    """
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        out_index = np.array(out_shape)
        in_index = np.array(in_shape)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            data = in_storage[in_pos]
            map_data = fn(data)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = map_data

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::
      fn_map = map(fn)
      fn_map(a, out)
      out
    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`
    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.
    Optimizations:
        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing
    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.
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
        out_index = np.array(out_shape, np.float32)
        a_index = np.array(a_shape, np.float32)
        b_index = np.array(b_shape, np.float32)
        for i in prange(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            a_data = a_storage[a_pos]
            b_data = b_storage[b_pos]
            map_data = fn(a_data, b_data)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = map_data

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function.
      fn_zip = zip(fn)
      c = fn_zip(a, b)
    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over
    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.
    Optimizations:
        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables
    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out
    Returns:
        None : Fills in `out`
    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        reduce_index = np.array(a_shape[reduce_dim], np.float32)
        reduce_shape = [a_shape[d] if d == reduce_dim else 1 for d in range(len(a_shape))]
        reduce_size = a_shape[reduce_dim]
        out_index = np.array(out_shape, np.float32)
        reduce_index = np.array(reduce_shape, np.float32)
        for i in prange(len(out)):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            for s in prange(reduce_size):
                to_index(s, reduce_shape, reduce_index)
                a_index = out_index + reduce_index
                a_pos = index_to_position(a_index, a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::
      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)
    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce
    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(
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
    """
    NUMBA tensor matrix multiply function.
    Should work for any tensor shapes that broadcast as long as ::
        assert a_shape[-1] == b_shape[-2]
    Optimizations:
        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.
    Args:
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
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for p in prange(len(out)):
        out_index = np.zeros(len(out_shape), np.int32)
        to_index(p, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)

        a_index = np.zeros(len(a_shape), np.int32)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        a_index[-2] = out_index[-2]

        b_index = np.zeros(len(b_shape), np.int32)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        b_index[-1] = out_index[-1]

        sum_out = 0
        for k in range(a_shape[-1]):
            a_index[-1] = k
            a_start = index_to_position(a_index, a_strides)
            b_index[-2] = k
            b_start = index_to_position(b_index, b_strides)
            sum_out += a_storage[a_start] * b_storage[b_start]
        out[out_pos] = sum_out


def matrix_multiply(a, b):
    """
    Batched tensor matrix multiply ::
        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]
    Where n indicates an optional broadcasted batched dimension.
    Should work for tensor shapes of 3 dims ::
        assert a.shape[-1] == b.shape[-2]
    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b
    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply


# import numpy as np
# from .tensor_data import (
#     to_index,
#     index_to_position,
#     broadcast_index,
#     shape_broadcast,
# )
# from numba import njit, prange
# import minitorch


# # TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# # This code will JIT compile fast versions your tensor_data functions.
# # If you get an error, read the docs for NUMBA as to what is allowed
# # in these functions.
# to_index = njit(inline="always")(to_index)
# index_to_position = njit(inline="always")(index_to_position)
# broadcast_index = njit(inline="always")(broadcast_index)


# def tensor_map(fn):
#     """
#     NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

#     Optimizations:

#         * Main loop in parallel
#         * All indices use numpy buffers
#         * When `out` and `in` are stride-aligned, avoid indexing

#     Args:
#         fn: function mappings floats-to-floats to apply.
#         out (array): storage for out tensor.
#         out_shape (array): shape for out tensor.
#         out_strides (array): strides for out tensor.
#         in_storage (array): storage for in tensor.
#         in_shape (array): shape for in tensor.
#         in_strides (array): strides for in tensor.

#     Returns:
#         None : Fills in `out`
#     """
#     def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
#         # Better for JIT
#         # out_index = out_shape.copy()
#         # in_index = in_shape.copy()
#         # Better for non-JIT
#         out_index = np.array(out_shape)
#         in_index = np.array(in_shape)
#         for i in range(len(out)):
#             to_index(i, out_shape, out_index)
#             broadcast_index(out_index, out_shape, in_shape, in_index)
#             in_pos = index_to_position(in_index, in_strides)
#             data = in_storage[in_pos]
#             map_data = fn(data)
#             out_pos = index_to_position(out_index, out_strides)
#             out[out_pos] = map_data

#     return njit(parallel=True)(_map)



# def map(fn):
#     """
#     Higher-order tensor map function ::

#       fn_map = map(fn)
#       fn_map(a, out)
#       out

#     Args:
#         fn: function from float-to-float to apply.
#         a (:class:`Tensor`): tensor to map over
#         out (:class:`Tensor`): optional, tensor data to fill in,
#                should broadcast with `a`

#     Returns:
#         :class:`Tensor` : new tensor
#     """

#     # This line JIT compiles your tensor_map
#     f = tensor_map(njit()(fn))

#     def ret(a, out=None):
#         if out is None:
#             out = a.zeros(a.shape)

#         # The pre-existing implementation
#         f(*out.tuple(), *a.tuple())

#         # Passes some JIT tests, very slowly
#         # _, out_size, out_stride = out.tuple()
#         # _, a_size, a_stride = a.tuple()
#         # out_size = out_size.astype(float)
#         # out_stride = out_stride.astype(float)
#         # a_size = a_size.astype(float)
#         # a_stride = a_stride.astype(float)
#         # f(out._tensor._storage, out_size, out_stride, a._tensor._storage, a_size, a_stride)
#         return out

#     return ret


# def tensor_zip(fn):
#     """
#     NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


#     Optimizations:

#         * Main loop in parallel
#         * All indices use numpy buffers
#         * When `out`, `a`, `b` are stride-aligned, avoid indexing

#     Args:
#         fn: function maps two floats to float to apply.
#         out (array): storage for `out` tensor.
#         out_shape (array): shape for `out` tensor.
#         out_strides (array): strides for `out` tensor.
#         a_storage (array): storage for `a` tensor.
#         a_shape (array): shape for `a` tensor.
#         a_strides (array): strides for `a` tensor.
#         b_storage (array): storage for `b` tensor.
#         b_shape (array): shape for `b` tensor.
#         b_strides (array): strides for `b` tensor.

#     Returns:
#         None : Fills in `out`
#     """
#     def _zip(
#         out,
#         out_shape,
#         out_strides,
#         a_storage,
#         a_shape,
#         a_strides,
#         b_storage,
#         b_shape,
#         b_strides,
#     ):
#         # Better for JIT
#         # out_index = out_shape.copy()
#         # a_index = a_shape.copy()
#         # b_index = b_shape.copy()
#         # Prior implementation
#         out_index = np.array(out_shape, np.float32)
#         a_index = np.array(a_shape, np.float32)
#         b_index = np.array(b_shape, np.float32)
#         for i in prange(len(out)):
#             to_index(i, out_shape, out_index)
#             broadcast_index(out_index, out_shape, a_shape, a_index)
#             broadcast_index(out_index, out_shape, b_shape, b_index)
#             a_pos = index_to_position(a_index, a_strides)
#             b_pos = index_to_position(b_index, b_strides)
#             a_data = a_storage[a_pos]
#             b_data = b_storage[b_pos]
#             map_data = fn(a_data, b_data)
#             out_pos = index_to_position(out_index, out_strides)
#             out[out_pos] = map_data

#     return njit(parallel=True)(_zip)


# def zip(fn):
#     """
#     Higher-order tensor zip function.

#       fn_zip = zip(fn)
#       c = fn_zip(a, b)

#     Args:
#         fn: function from two floats-to-float to apply
#         a (:class:`Tensor`): tensor to zip over
#         b (:class:`Tensor`): tensor to zip over

#     Returns:
#         :class:`Tensor` : new tensor data
#     """
#     f = tensor_zip(njit()(fn))

#     def ret(a, b):
#         c_shape = shape_broadcast(a.shape, b.shape)
#         out = a.zeros(c_shape)
    
#         # The pre-existing implementation
#         f(*out.tuple(), *a.tuple(), *b.tuple())

#         # Passes more JIT tests, but very slowly
# #         _, out_size, out_stride = out.tuple()
# #         _, a_size, a_stride = a.tuple()
# #         _, b_size, b_stride = b.tuple()

# #         out_size = out_size.astype(float)
# #         out_stride = out_stride.astype(float)
# #         a_size = a_size.astype(float)
# #         a_stride = a_stride.astype(float)
# #         b_size = b_size.astype(float)
# #         b_stride = b_stride.astype(float)

# #         f(out._tensor._storage, out_size, out_stride, a._tensor._storage, a_size, a_stride, b._tensor._storage, b_size, b_stride)
#         return out

#     return ret


# def tensor_reduce(fn):
#     """
#     NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

#     Optimizations:

#         * Main loop in parallel
#         * All indices use numpy buffers
#         * Inner-loop should not call any functions or write non-local variables

#     Args:
#         fn: reduction function mapping two floats to float.
#         out (array): storage for `out` tensor.
#         out_shape (array): shape for `out` tensor.
#         out_strides (array): strides for `out` tensor.
#         a_storage (array): storage for `a` tensor.
#         a_shape (array): shape for `a` tensor.
#         a_strides (array): strides for `a` tensor.
#         reduce_dim (int): dimension to reduce out

#     Returns:
#         None : Fills in `out`

#     """
     
#     def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
#         # Better for JIT
#         # reduce_index = a_shape[reduce_dim] + 1 - 1 # these matter for JIT
#         # reduce_shape = np.array([a_shape[d] if d == reduce_dim else 1 for d in range(len(a_shape))]).copy()
#         # reduce_size = int(a_shape[reduce_dim] + 1 - 1)
#         # out_index = out_shape.copy()
#         # reduce_index = reduce_shape.copy()
        
#         # Prior implementation
#         reduce_index = np.array(a_shape[reduce_dim], np.float32)
#         reduce_shape = [a_shape[d] if d == reduce_dim else 1 for d in range(len(a_shape))]
#         reduce_size = a_shape[reduce_dim]
#         out_index = np.array(out_shape, np.float32)
        
#         for i in prange(len(out)):
#             to_index(i, out_shape, out_index)
#             out_pos = index_to_position(out_index, out_strides)
#             for s in prange(reduce_size):
#                 to_index(s, reduce_shape, reduce_index)
#                 a_index = out_index + reduce_index
#                 a_pos = index_to_position(a_index, a_strides)
#                 out[out_pos] = fn(out[out_pos], a_storage[a_pos])

#     return njit(parallel=True)(_reduce)


# def reduce(fn, start=0.0):
#     """
#     Higher-order tensor reduce function. ::

#       fn_reduce = reduce(fn)
#       out = fn_reduce(a, dim)


#     Args:
#         fn: function from two floats-to-float to apply
#         a (:class:`Tensor`): tensor to reduce over
#         dim (int): int of dim to reduce

#     Returns:
#         :class:`Tensor` : new tensor
#     """

#     f = tensor_reduce(njit()(fn))

#     def ret(a, dim):
#         out_shape = list(a.shape)
#         out_shape[dim] = 1

#         # Other values when not sum.
#         out = a.zeros(tuple(out_shape))
#         out._tensor._storage[:] = start


#         # The pre-existing implementation
#         f(*out.tuple(), *a.tuple(), *b.tuple())

#         # Passes more JIT tests, but very slowly
# #         _, out_size, out_stride = out.tuple()
# #         _, a_size, a_stride = a.tuple()

# #         out_size = out_size.astype(float)
# #         out_stride = out_stride.astype(float)
# #         a_size = a_size.astype(float)
# #         a_stride = a_stride.astype(float)

# #         f(out._tensor._storage, out_size, out_stride, a._tensor._storage, a_size, a_stride, dim)
#         return out

#     return ret


# @njit(parallel=True, fastmath=True)
# def tensor_matrix_multiply(
#     out,
#     out_shape,
#     out_strides,
#     a_storage,
#     a_shape,
#     a_strides,
#     b_storage,
#     b_shape,
#     b_strides,
# ):
#     """
#     NUMBA tensor matrix multiply function.

#     Should work for any tensor shapes that broadcast as long as ::

#         assert a_shape[-1] == b_shape[-2]

#     Optimizations:

#         * Outer loop in parallel
#         * No index buffers or function calls
#         * Inner loop should have no global writes, 1 multiply.


#     Args:
#         out (array): storage for `out` tensor
#         out_shape (array): shape for `out` tensor
#         out_strides (array): strides for `out` tensor
#         a_storage (array): storage for `a` tensor
#         a_shape (array): shape for `a` tensor
#         a_strides (array): strides for `a` tensor
#         b_storage (array): storage for `b` tensor
#         b_shape (array): shape for `b` tensor
#         b_strides (array): strides for `b` tensor

#     Returns:
#         None : Fills in `out`
#     """
#     a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
#     b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

#     for p in prange(len(out)):
#         out_index = np.zeros(len(out_shape), np.int32)
#         to_index(p, out_shape, out_index)
#         out_pos = index_to_position(out_index, out_strides)

#         a_index = np.zeros(len(a_shape), np.int32)
#         broadcast_index(out_index, out_shape, a_shape, a_index)
#         a_index[-2] = out_index[-2]

#         b_index = np.zeros(len(b_shape), np.int32)
#         broadcast_index(out_index, out_shape, b_shape, b_index)
#         b_index[-1] = out_index[-1]

#         sum_out = 0
#         for k in range(a_shape[-1]):
#             a_index[-1] = k
#             a_start = index_to_position(a_index, a_strides)
#             b_index[-2] = k
#             b_start = index_to_position(b_index, b_strides)
#             sum_out += a_storage[a_start] * b_storage[b_start]
#         out[out_pos] = sum_out


# def matrix_multiply(a, b):
#     """
#     Batched tensor matrix multiply ::

#         for n:
#           for i:
#             for j:
#               for k:
#                 out[n, i, j] += a[n, i, k] * b[n, k, j]

#     Where n indicates an optional broadcasted batched dimension.

#     Should work for tensor shapes of 3 dims ::

#         assert a.shape[-1] == b.shape[-2]

#     Args:
#         a (:class:`Tensor`): tensor data a
#         b (:class:`Tensor`): tensor data b

#     Returns:
#         :class:`Tensor` : new tensor data
#     """

#     # Make these always be a 3 dimensional multiply
#     both_2d = 0
#     if len(a.shape) == 2:
#         a = a.contiguous().view(1, a.shape[0], a.shape[1])
#         both_2d += 1
#     if len(b.shape) == 2:
#         b = b.contiguous().view(1, b.shape[0], b.shape[1])
#         both_2d += 1
#     both_2d = both_2d == 2

#     ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
#     ls.append(a.shape[-2])
#     ls.append(b.shape[-1])
#     assert a.shape[-1] == b.shape[-2]
#     out = a.zeros(tuple(ls))

#     tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

#     # Undo 3d if we added it.
#     if both_2d:
#         out = out.view(out.shape[1], out.shape[2])
#     return out


# class FastOps:
#     map = map
#     zip = zip
#     reduce = reduce
#     matrix_multiply = matrix_multiply

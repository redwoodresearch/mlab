# Tips and Tricks for Day 1

## Elementwise Operations on Tensors

If `a` and `b` are tensors, then the expression `a and b` does NOT do what you might expect, which is elementwise logical AND. The correct way to spell this in NumPy or PyTorch is `a & b`, or `a | b` for elementwise logical OR.

Be careful with operator precedence too: `a > 0 & a < 10` does NOT do what you expect. The correct way is `(a > 0) & (a < 10)`.

You can do elementwise logical NOT with `~a`.

## Tensor Methods vs Tensor Functions vs Tensor Operators

For a tensor `a`, it's equivalent to do `torch.exp(a)` and `a.exp()`. Both allocate a new output tensor containing `e^x` elementwise. 

An important difference is that `torch.exp` can optionally take an `out` tensor as an additional argument, in which case no allocation is done and the result is written directly to `out`. 

When operators exist, be careful that you understand what function or method they are equivalent to. 

For example for tensors `a`, `b`, the operation `a + b` is equivalent to `a.add(b)` (allocates and returns a new tensor).

Gotcha: the operation `a += b` is NOT equivalent to `a = a.add(b)` which would allocate a new tensor and release the old one, but a special method `a.add_(b)` (note the underscore) which modifies tensor `a` in-place. 

In sme cases it's important to know when you are modifying things in-place. When multiple pieces of code have references to the same tensor, they all see changes from any in-place operation but wouldn't see changes if you just create a new tensor. For example, when your model and your optimizer both have a reference to the same model parameter.

Sometimes, you cannot backpropagate through an in-place operation. Typically you will see a helpful error message when this is the case and you try to call backward().


## Einops.Reduce

You can pass a callable to einops.reduce but that callable must take as arguments (tensor, tuple of axes to reduce over). 

This means some functions like `torch.all` or `torch.any` that have a different signature don't work with einops.reduce. 

However, you can use `torch.amin` or `torch.amax` with einops.reduce to compute the desired result.


## Tuple Unpacking Idiom

A useful idiom in Python is the following, for some tensor `a`:

`B, T, C = a.shape`

This creates three variables with integer values equal to the length of `a` along each dimension. 

This is a self-documenting way to say that you expect `a` to be a tensor with three dimensions, and the variable names can indicate something like the three dimensions are batch, time, and channel dimentions.

It also raises an error immediately if `a` does not have exactly 3 dimensions, which is a great way to prevent your code from silently doing the wrong thing.
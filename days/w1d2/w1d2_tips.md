# Tips and Tricks for W1D2

## nn.Sequential vs nn.ModuleList vs list of nn.Module

When writing a `nn.Module` that contains other modules as children, be sure to understand the implications of using each of the following options:

1) list of `nn.Module`

```python
class MyModule(nn.Module):
    def __init__(self):
        self.layers = [Module1(), Module2()]
```

Using a simple list, the parent `Module` won't recognize that these are child modules. The child modules won't be saved or loaded with the parent, and the child parameters won't show up in `parent.parameters()`. This behavior is rarely what you want.

2) `nn.ModuleList`

```python
class MyModule(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([Module1(), Module2()])
```

Using a ModuleList, the parent Module does recognize these as child modules.


3) `nn.Sequential`

```python
class MyModule(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential(Module1(), Module2())

        # equivalently:
        modules = [Module1(), Module2()]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
```

Using a Sequential, the parent Module does recognize these as child modules. Additionally, you can call the sequential and it will act as a pipeline - your input is passed to the first module, then the first module's output is passed to the second module, etc and the Sequential returns the output of the las module.

Note that the syntax differs - while a ModuleList accepts a single list argument, a Sequential takes each module as a separate argument. You can use the sequence unpacking operator `*` as shown to split a list into separate arguments.


## Register_buffer

A `nn.Module` can call `self.register_buffer` when you want a specific tensor to be saved and loaded along with the model, but you don't want that tensor to receive the special treatment that a `nn.Parameter` does. (A `nn.Parameter` requires grad by default, and shows up in `model.parameters()`).

An example of when you would use this is storing the running mean and variance of BatchNorm layer.

## torch.is_contiguous

Normally when we think of a contiguous tensor, this just means all the elements of the tensor are stored in one solid block of memory. `torch.tensor.is_contiguous` is more strict than this - by default it also requires that the memory format be row-major (like a C style array). 

In particular, if `a` is contiguous and row-major then `a.T.is_contiguous` is `False` for this reason, even though `a.T` is still one block of memory.


## Built-in Name Shadowing

Try not to create variables with the same names as Python built-in functions, because Python won't stop you and this can cause hard to understand bugs later on in that scope. Example:

```python
len = my_tensor.shape[0]  # Legal but very naughty!

...

# Potentially hundreds of lines later
# Looks innocent but throws an exception because len is an int now
size = len(my_array)  
```

## Thinking about Multiple Convolutional Kernels

The shapes in a convolutional layer can be confusing when there are multiple kernels being applied together. For a 1D convolution, shapes go like this:

```
input: (batch, channels_in, kernel_width)
kernels: (channels_out, channels_in, kernel_width)
output: (batch, channels_out, output_width)
```

You can always unfold tensor operations into a nested for loop. The following isn't exactly right because of padding and such but just shows the dimensions:

```python

for b in range(batch_size):
    for channel_out in range(channels_out_size):
        for channel_in in range(channels_in_size):
            for w in range(output_width):
                for k in range(kernel_width):
                    output[b, channel_out, w] += input[b, channel_in, w + k] * kernel[channel_out, channel_in, k]
```

This shows that each batch element and each output channel are independent. A single output channel at a single output position sums over all the input channels and k spatial positions.
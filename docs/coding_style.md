# Words of "Wisdom" from Old Man Chris

This document is intended to contain tips and tricks and good habits to get into during and after the bootcamp. 

Good style and best practices are always a contentious topic, and the Internet and popular books are full of advice that is either subjective preference disguised as fact, or advice mainly beneficial to working on million line codebases.

The following are my personal opinions after 15 years of Python development and countless errors. Try doing things my way first, then try doing them the opposite way and form your own opinions!


## Jupyter Workflow

Write code in the smallest units possible, at the top level of the notebook. This allows you to easily inspect each step to make sure it works before you assemble the pieces into a function. Don't just write a big chunk of code and then hope it works.

If you're working on a function, try to execute and test the function in the same block as the function. This prevents the confusing situation where you change the function, then forget to re-run the function definition.

Print or use our included `utils.tpeek` helper to see expressions inside functions.

```python
def my_func(x):
    temp = x**2
    tpeek(temp)
    temp2 = temp + 1
    return temp2

my_func(5)
```

Run your cells in order to prevent confusing things happening. You can use "run cell and all below" command. Periodically restart your kernel and "run all" to make sure everything is working properly.

Learn your notebook keyboard shortcuts! At minimum, be able to rapidly add cells above or below, delete, cut and paste cells, and convert cells to code/markdown.

## Jupyter Notebooks - Markdown

Learn [Markdown syntax](https://www.markdownguide.org/basic-syntax/) and use it in your notebooks:
- Use # to create headers - keeps notebooks organized
- Prefer longer comments in Markdown cells instead of multi-line code comments
- Display equations using LaTeX expressions by surrounding them with dollar signs. `$\frac{1}{2}$` renders as $\frac{1}{2}$.

## Jupyter Notebooks - The Mighty Semicolon

Use a semicolon at the end of a line to prevent output from appearing.  

```python
# Messy - prints "[<matplotlib.lines.Line2D at 0x1379e8250>]"
plt.plot([1, 2, 3, 4])  
```

```python
# Tidy - no useless output
plt.plot([1, 2, 3, 4]);
```

Bonus: if your cell prints a lot of output and you want to suppress all of it, put `%%capture` at the top of the cell.


## PyTorch - Device Agnostic Code

Unless you have a reason not to, write code that will use the GPU when available and otherwise use the CPU. A good way to do this is:

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Later in the file...
my_tensor = torch.zeros(..., device=DEVICE)
my_other_tensor = torch.rand_like(my_tensor)
my_model.to(DEVICE)
```

Note the functions `zeros_like`, `rand_like`, and similar create a new tensor using the same device as the input argument. This is a concise way to avoid passing `DEVICE` explicitly. 

## PyTorch - nn.Sequential vs Module.forward()

Consider the best approach for each situation. Each has their advantages.

```python

# Approach 1: write out forward pass explicitly. Most expressive option.
class MyModule(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(...)
        self.bn = nn.BatchNorm(...)
        self.maxpool = nn.MaxPool2D(...)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x

# Approach 2: use nn.Sequential. More compact, easy to modify.
my_module = nn.Sequential(
    nn.Conv2D(...),
    nn.BatchNorm(...)
    nn.ReLU(...)
    nn.MaxPool2D(...)
)
```

## Keyword Arguments

Prefer calling functions using keyword arguments when there are more than a couple arguments, especially if there are multiple arguments of the same type. 

```python
# Harder to read:
# What do these arguments mean? 
# Are the units on the first argument degrees or radians or something else? 
# Are the second and third arguments in the correct order?
torchvision.transforms.RandomAffine(10, (0.05, 0.05), (0.95, 1.05), 5)

# Easier to read:
torchvision.transforms.RandomAffine(
    degrees=10, 
    translate=(0.05, 0.05), 
    scale=(0.95, 1.05), 
    shear=5)
```

## Imports

Prefer placing all imports at the top of the notebook or file. This helps readers understand what libraries they need installed, and makes it easy to reorder code without accidentally putting calls before the needed import.

Prefer using aliases to reduce typing. The following are standard aliases that others will recognize.

```python
import torch as t
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
import tensorflow as tf
import statsmodels.api as sm
import xgboost as xgb
```

Use `from module import *` with extreme caution. Sooner or later you will accidentally call a different function than you were intending, and it will silently do the wrong thing. Make sure you understand what is happening below:

```
>>> start = -1
>>> sum(range(5), start)
9
>>> from numpy import *
>>> sum(range(5), start)
10
```

## Einops

Prefer the functions in the `einops` library where possible. [Their documentation](https://github.com/arogozhnikov/einops#why-use-einops-notation-) has a list of benefits from using the library.

## Jupyter Notebooks - Special Commands

In a code cell, you can use the prefix `!` to execute a command in the shell.

```python
!pip install black
```

You can use `??` after an expression to view its documentation.

```python
torch.randn??
```

## Autoreload - Using Notebooks and Scripts Together

Notebooks are great for exploration and for small amounts of code, but using `.py` files is superior for larger amounts of code - they are easier to test, easier to version control, and easier to reuse.

A great way to transition from notebooks to notebooks + files is to start placing functions in files and use the [`autoreload`](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html) command in your notebook to automatically reload your file at every script execution.

```python
%load_ext autoreload
%autoreload 1  # enable autoreload

%aimport my_file

# Now changes to my_file are automatically picked up when you re-run a cell
my_file.my_function() 
```

## Jupyter Notebooks - Benchmarking

The %timeit command will repeatedly execute an expression and measure the time it takes. It can be very non-obvious how fast or slow things are when it comes to vectorized code, so make predictions and notice confusion if your predictions are wrong!

```python
x = torch.arange(1000 * 1000).reshape((-1, 100))
y = x.T.clone()
%timeit x @ y
%timeit x.T @ y.T

3.87 s ± 39.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
47.6 ms ± 131 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
x = torch.arange(1000 * 1000)

%timeit x ** 4
%timeit (x ** 2) ** 2
%timeit x * x * x * x

1.32 ms ± 45.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
2.33 ms ± 261 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
2.53 ms ± 180 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

This applies to code running on CPU. It's more challenging to properly benchmark GPU code.


## Jupyter Notebooks - special variables

In a notebook or IPython, the special variable `_` refers to the last result evaluated. Also, each cell has a number like `[18]`, and the special variable `_18` refers to that result of that cell.

This is really useful if you realize after running a long computation that you want to use the output, but you forgot to assign the result to a variable beforehand.


## Image Conventions

Be aware of the ordering of dimensions in a tensor. For a tensor containing a batch of images, PyTorch code usually uses the `(batch, channels, height, width)` ordering, whereas Tensorflow code usually uses the `(batch, height, width, channels)` ordering.

Be aware of preprocessing steps expected by a model. The model may still work without the right preprocessing, but with reduced accuracy. For example, many image models expect image data to be converted from 0-255 pixel data to floats in the range [0,1] and then normalized using statistics from ImageNet.


## Debugging Notebooks

Recent versions of VS Code support running the VS Code debugger on notebooks. I most commonly use this to step into library code and figure out where an exception is coming from. You will need to add `"justMyCode": false` to your `launch.json` to allow stepping into library code.

Another way to attach a debugger is by using the command `%pdb 1` in a cell. This will automatically launch the `pdb` debugger when an exception is encountered. From here you can evaluate expressions and press `c` to continue execution. It's worth learning `pdb` because you can use it in many situations like Colab or a remote SSH session where better tools aren't available to you.

Stepping through unfamiliar code is a good way to learn. You can see the flow of execution and what types and values are normally available.

## Progress Bars with tqdm

`tqdm` is a library for showing progress bars, which is very useful for long-running tasks. It can be used in the [command line](https://tqdm.github.io/docs/tqdm/) or a [notebook](https://tqdm.github.io/docs/notebook/).

## Assertions

Assertions ensure that assumptions you've made in your code are correct. It's always better to find problems early in execution instead of much later or worse, silently computing the wrong result.

```python
assert image.height == image.width, "Expected a square image!"
assert arr.ndim == 3, "Array must be three-dimensional!"
assert arr.min() >= 0, "Array must be all non-negative!"
```

When learning, use assertions to solidify your understanding of what functions are doing.

```python
t = torch.randn((100, 100))
# Hm, what does a negative dim do exactly? 
out = torch.nn.functional.softmax(dim=-1)

# I think this should be all ones - is it?
sums = out.sum(dim=1)
assert torch.allclose(sums, torch.ones_like(sums)))
```

It can be hard to assert on the contents of tensors when they have many elements. You can ask yourself - do I know what the mean or std of this tensor should be along some slice? Should the elements all be between some interval? 


## Shape Unpacking

When working with multi-dimensional arrays, a useful pattern is:

```python
B, C, H, W = my_tensor.shape
```

This raises an exception if my_tensor does not have exactly 4 dimensions, which helps to catch incorrect input early. It also gives easy and readable access to the lengths of each dimension. 


## Matplotlib - Default Styles

Prefer to set default styles at the top of the notebook. This makes them easy to change in one place rather than scattered through the notebook.

```python
# Less preferred:
fig1 = plt.figure(figsize=(12, 12))
def my_plot_fn(figsize=(12, 12)):
    fig2, ax = plt.subplots(figsize=figsize)
    ...


# Preferred:
plt.rcParams["figure.figsize"] = (12, 12)
fig1 = plt.figure()
def my_plot_fn():
    fig2, ax = plt.subplots()
```

## Matplotlib - Object Oriented Interface

Prefer creating figure and axes objects and calling their methods to using `pyplot` functions that mutate global state. This reduces bugs and performance problems when you have many figures and axes in one program or notebook.

```python
# Less preferred:
plt.subplot(211)
plt.plot([1, 2, 3, 4])
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.subplot(212)
plt.plot([2, 3, 4, 5])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title')

# Preferred:
fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot([1, 2, 3, 4])
ax1.set(xlim=[0,5], ylim=[0,5])
ax2.plot([2, 3, 4, 5])
ax2.set(xlabel='X axis', ylabel='Y axis', title='Title')
```

## Code Formatting

We use the formatting tool [black](https://black.readthedocs.io/en/stable/) for formatting source code. This reduces the size of diffs by ensuring formatting is consistent throughout a code base.

In Visual Studio Code, add `"editor.formatOnSave": true` to your `settings.json` to automatically format your source code when you save a source file.

In Visual Studio notebooks, you can use the command `Format Cell` or its keyboard shortcut to format an individual cell. 

## Serialization

There are a huge number of ways to save data, with various pros and cons. For now, we'll keep it simple:

- For small arrays, `np.savetxt` and `np.loadtxt` are nice because the output is plain text. It's portable, and easy to check that you saved it properly.
- For torch modules, prefer `torch.save(module.state_dict()) ` instead of saving entire module. See [the docs](https://pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-torch-nn-modules) for detail on why.
- `pickle.dump` is fine for temporary use, but there's usually a better alternative for serious work.

# Useful Topics

## Floating Point Numbers

This is a large topic. The short list of things to know are:

- Use [`torch.allclose`](https://pytorch.org/docs/stable/generated/torch.allclose.html) or similar to compare arrays of floats.
- Floats can represent infinity `float('inf')` and negative infinity `float(-inf)`. A common way to get an infinity is to divide a non-zero number by zero.
- Floats can represent a special value called not a number `float('NaN')`. A common way to get a NaN is to divide zero by zero. NaN is the only number that is not equal to itself, so you must use `np.isnan()` or similar to test for it. 
- Floats have both positive zero, and negative zero `float('-0')`. Positive and negative zeros compare equal, so usually you won't need to care about signed zeros.
- You may see functions whose only purpose is to do the right thing with floating points, where the "obvious" way actually does the wrong thing. An example of this would be [`np.logaddexp`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html).
- If you suspect something is not working because of floating point weirdness, just ask your TA.

[What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) is a classic article if you want to really get into the weeds.

## Math for Deep Learning

For using deep learning models you don't need a huge amount of math, as a lot is already done for you in libraries. If you know basic calculus you can get pretty far by applying calculus rules to individual elements of a tensor, combined with experimenting until something works.

A good starting point to dig in if you want to understand scientific papers is [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/).

## Version Control

Git is the most commonly used version control system, and you will be expected to know it in most jobs. 

Git is very unintuitive, but since most people use it you will need to know at least the basics to collaborate with them. I still get confused and have to Google whenever I want to do something out of the ordinary with Git.

Even for code that other developers never see, I find it useful to see what changes I've done since the last commit, and to be able to look at past commits and understand why I made a certain change. 

[Git Immersion](https://gitimmersion.com/index.html) is a nice way to get started with Git.
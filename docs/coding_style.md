# Coding Style

There are many subjective arguments about what constitutes "good" coding style. We will follow the standard [PEP8](https://www.python.org/dev/peps/pep-0008/) plus some additional advice. 

The following are good habits to get into, but don't stress about them since the code you are writing for this bootcamp likely won't be maintained and read by others in the future.

## Keyword Arguments

Prefer calling functions with keyword arguments when there are more than a couple arguments, or if there are multiple arguments of the same type. 

```python
# Less preferred: 
# What do these arguments mean? 
# Are the units on the first argument degrees or radians or something else? 
# Are the second and third arguments in the correct order?
torchvision.transforms.RandomAffine(10, (0.05, 0.05), (0.95, 1.05), 5)

# Preferred: 
torchvision.transforms.RandomAffine(
    degrees=10, 
    translate=(0.05, 0.05), 
    scale=(0.95, 1.05), 
    shear=5)
```

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

## Jupyter Notebooks - Workflow

Write and debug code in the smallest units possible, at the top level of the notebook. This allows you to easily inspect each step to make sure it works before you assemble the pieces into a function.

## Jupyter Notebooks - Markdown

Markdown cells have many uses:
- Use # to create headers and organize your notebooks
- Prefer longer explanations in Markdown cells instead of multi-line code comments
- Display equations using LaTeX expressions by surrounding them with dollar signs as in: `$\frac{1}{2}$`

## Jupyter Notebooks - The Mighty Semicolon

By default, executing a cell prints out the value of the last line in the cell. However, you can prevent this by ending that line with a semicolon. 

```python
# Less preferred
# Prints "[<matplotlib.lines.Line2D at 0x1379e8250>]"
plt.plot([1, 2, 3, 4])  
```

```python
# Preferred - no useless output
plt.plot([1, 2, 3, 4]);
```

## Jupyter Notebooks - Special Commands

In a code cell, you can use the prefix `!` to execute a command in the shell.

```python
!pip install black
```

You can use `??` after an expression to view its documentation.

```python
torch.randn??
```

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

## Einops

Prefer the functions in the `einops` library where possible. [Their documentation](https://github.com/arogozhnikov/einops#why-use-einops-notation-) has a list of benefits from using the library.


## Code Formatting

We use the formatting tool [black](https://black.readthedocs.io/en/stable/) for formatting source code. This reduces the size of diffs by ensuring formatting is consistent throughout a code base.

In Visual Studio Code, add `"editor.formatOnSave": true` to your `settings.json` to automatically format your source code when you save a source file.

In Visual Studio notebooks, you can use the command `Format Cell` or its keyboard shortcut to format an individual cell. 

## Version Control

Git is the most commonly used system and well worth learning for any programmer, even if you primarily work alone. Teaching Git is beyond the scope of this course, but we recommend you learn the basics on your own time.




### Misc Notes / WIP

`torch.set_printoptions` and `numpy.set_printoptions` are occasionally useful. 


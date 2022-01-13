from torch.nn.functional import conv1d as torch_conv1d
from days.utils import is_equal_test

## Test 1

input = torch.tensor([[[1, 2, 3, 4]]])
weights = torch.tensor([[[0, 0]]])  # TODO: replace this
output = torch_conv1d(input, weights)
expected = torch.tensor([7.0, 11.0, 15.0])
is_equal_test(output=output, expected=expected)

## Test 2

input = torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1]]])  # TODO: replace this
weights = torch.tensor([[[2, 0, -2]]])
output = torch_conv1d(input, weights)
expected = torch.tensor([[[210, 30, -12, -4, -4, -4]]])
is_equal_test(output=output, expected=expected)

## Test 3

input = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [2, 3, 2]]])
weights = torch.tensor([[[0], [0]]])  # TODO: replace this
output = torch_conv1d(input, weights)
expected = torch.tensor([[[9, 12, 15]], [[5, 7, 5]]])
is_equal_test(output=output, expected=expected)

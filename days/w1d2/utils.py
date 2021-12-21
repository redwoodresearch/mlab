
def is_equal_test(*, output, expected, test_name='Test'):
  successful = torch.allclose(expected.to(float), output.to(float))
  if successful:
    print(f'{test_name} passed!')
  else:
    print(f'{test_name} failed')
    print(f'Output:\n{output}')
    print(f'Expected:\n{expected}')


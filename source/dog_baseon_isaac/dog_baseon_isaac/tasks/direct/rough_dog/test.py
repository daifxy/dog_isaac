import torch

a = torch.tensor([1.0, 2.0, 3.0])
e = torch.tensor([True, False, True])
b = a
# a[0] = 10.0
print(a, b)

c = a.clone()
# a[1] = 20.0
a = torch.clip(a-2, 0, 4)
print(e.sum(dim=0))
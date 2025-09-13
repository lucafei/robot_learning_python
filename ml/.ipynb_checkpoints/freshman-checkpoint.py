import torch
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms

# Basic tensor operations in PyTorch
x = torch.arange(12)
print(x)
print(x.shape)
x = x.reshape(3, 4)
print(x.shape)
print(x)
x = torch.zeros(3, 4)
print(x)
# operators
u = torch.tensor([1, 2, 3])
v = torch.tensor([4, 5, 6])
print(u+v)
print(u-v)
print(u*v)
print(u/v)
print(u**v)

# matrix operations
print(torch.exp(x))
c = torch.cat((u, v), dim=0)
print(c)

print(u==v)

print(u.sum())

a = torch.arange(12).reshape(3, 4)
a = torch.cat((a, a), dim=0)
print(a)
print(a.shape)

print(a[-1])
print(a[1:3])
a[1:3, 0] = 100
print(a)
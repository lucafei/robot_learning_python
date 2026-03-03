'''
Author: yinfei yinfei@minieye.cc
Date: 2025-07-31 10:00:38
LastEditors: yinfei yinfei@minieye.cc
LastEditTime: 2025-08-04 21:10:25
FilePath: /rl/ml/freshman.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import os

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

os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join(os.path.join('..', 'data', 'house_tiny.csv'))
with open(data_file, 'w') as f:
    f.write('NumRooms, Alleys, Price\n')
    f.write('NA, Pave, 127500\n')
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')

import pandas as pd
import numpy as np
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 只对数值列进行填充，跳过字符串列
numeric_inputs = inputs.select_dtypes(include=[np.number])
inputs = inputs.fillna(numeric_inputs.mean())
print(inputs)

inputs.dropna(axis=1, thresh=(inputs.count().min()+1))
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))

print(X)
print(y)


A = torch.arange(20).reshape(5, 4)
print(A)

print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)

print(B == B.T)

C = A.clone()
print(A+C)

print(A*C)

x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())
print(C.sum(dim=0))
print(C.sum(dim=0).shape)

print(C.sum(dim=[0, 1]))
print(C.sum(dim=0, keepdim=True).shape)

print(C.cumsum(dim=0))

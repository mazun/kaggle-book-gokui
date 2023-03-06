# 写経 https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/0_Learn%20the%20Basics/0_1_tensors_tutorial_js.ipynb#scrollTo=wHWwoi5wo65p

# %%
import torch
import numpy as np

# %%
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
x_data

# %%
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_np

# %%
x_ones = torch.ones_like(x_data)
x_ones

# %%
x_rand = torch.rand_like(x_data, dtype=torch.float)
x_rand

# %%
shape = (2, 3,)
rand_tensor = torch.rand(shape)
rand_tensor

# %%
ones_tensor = torch.ones(shape)
ones_tensor

# %%
zeros_tensor = torch.zeros(shape)
zeros_tensor

# %%
tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')

# %%
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# %%
print(f'Device tensor is stored on: {tensor.device}')

# %%
tensor = torch.rand(4, 4)
print('First row:', tensor[0])
print('First column:', tensor[:, 0])
print('Last column', tensor[..., -1])
print(all(tensor[:, -1] == tensor[..., -1]))
tensor[:, 1] = 0
print(tensor)

# %%
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# %%
ts = torch.stack([tensor, tensor, tensor], dim=1)
print(ts)

# %%
# 行列の積
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1 == y2)
print(y1)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1 == y3)

# 要素ごとの積
z1 = tensor * tensor
z2 = tensor.mul(tensor)
print(z1 == z2)

z3 = torch.rand_like(tensor)
torch.mul(z1, z2, out=z3)
print(z1 == z3)

# %%
agg = tensor.sum()
agg_item = agg.item()
print(agg, type(agg))
print(agg_item, type(agg_item))

# %%
print(tensor)
tensor.add_(5)
print(tensor)

# %%
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

# %%
t.add_(1)
print(f't: {t}')
print(f'n: {n}')  # also changes following to t

# %%
n = np.ones(5)
t = torch.from_numpy(n)
print(f't: {t}')
print(f'n: {n}')

# %%
np.add(n, 1, out=n)
print(f't: {t}')  # also changes following to n
print(f'n: {n}')

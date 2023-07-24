# %%
import torch

# %%
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# %%[markdown]
# ![graph](https://pytorch.org/tutorials/_images/comp-graph.png)
# ## Logits
# 正規化する前の値。softmaxをかけて正規化(=確率に変換)する。
# ## Cross Entropy
# 分類問題に使える損失関数。

# %%
print("Gradient function for z =", z.grad_fn)
print("Gradient function for loss =", loss.grad_fn)

# %%
loss.backward()
print(w.grad)
print(b.grad)

# %%

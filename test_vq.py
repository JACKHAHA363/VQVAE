import torch
from torch import nn
from torch.autograd import Variable
from model import VQLayer
import ipdb
K=3
D=2
bz = 5


x = Variable(torch.rand(bz, D), requires_grad=True)
vq = VQLayer(D, K)
y = vq(x)
z = torch.sum(y)
z.backward()

print("embs are", vq.emb.weight.data)
print("quantization", y.data)

# if emb_i is chosen for k times, then ith row should be all k
print("emb grads", vq.emb.weight.grad.data)

# should be all zeros
print("x grads", x.grad.data)

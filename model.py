import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb
class VectorQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, emb):
        """
        x: (bz, D)
        emb: (emb_num, D)
        output: (bz, D)
        """
        dist = row_wise_distance(x, emb)
        indices = torch.min(dist, -1)[1]
        ctx.indices = indices
        ctx.emb_num = emb.size(0)
        ctx.bz = x.size(0)
        return torch.index_select(emb, 0, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.indices.view(-1,1)
        bz = ctx.bz
        emb_num = ctx.emb_num

        # get a one hot index
        one_hot_ind = torch.zeros(bz, emb_num)
        one_hot_ind.scatter_(1, indices, 1)
        one_hot_ind = Variable(one_hot_ind, requires_grad=False)
        grad_emb = torch.mm(one_hot_ind.t(), grad_output)
        return grad_output, grad_emb


def row_wise_distance(m1, m2):
    """
    m1: (a,p) m2: (b,p)
    result:
    dist (a, b), where dist[i,j] = l2_dist(m1[i], m2[j])
    """
    a = m1.size(0)
    b = m2.size(0)
    mm1 = torch.stack([m1]*b).transpose(0,1)
    mm2 = torch.stack([m2]*a)
    return torch.sum((mm1-mm2)**2, 2).squeeze()


class VQLayer(nn.Module):
    def __init__(self, D, K):
        super(VQLayer, self).__init__()
        self.emb = nn.Embedding(K, D)
        self.K = K
        self.D = D

    def forward(self, x):
        """
        x: (bz, D)
        """
        return VectorQuantization.apply(x, self.emb.weight)


class VQVae(nn.Module):
    def __init__(self, enc, dec, emb_dim, emb_num):
        super(VQVae, self).__init__()
        self.enc = enc
        self.dec = dec
        self.vqlayer = VQLayer(D=emb_dim, K=emb_num)

    def forward(self, x):
        self.z_e = self.enc(x)
        self.z_q = self.vqlayer(self.z_e)
        self.x_reconst = self.dec(self.z_q)
        return self.x_reconst

    def sample_from_modes(self):
        """
        sample from the discrete representation
        """
        zq = self.vqlayer.emb.weight
        samples = self.dec(zq)
        return samples


class MLEenc(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(MLEenc, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, emb_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1)


class MLEdec(nn.Module):
    def __init__(self, emb_dim, input_dim):
        super(MLEdec, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 400)
        self.fc2 = nn.Linear(400, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(h))

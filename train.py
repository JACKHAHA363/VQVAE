from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VQLayer, MLEdec, MLEenc, VQVae
import ipdb

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--emb-dim', default=500, type=int)
parser.add_argument('--emb-num', default=10, type=int)
parser.add_argument('--beta', default=0.25, type=float)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


enc = MLEenc(784, args.emb_dim)
dec = MLEdec(args.emb_dim, 784)
vqvae = VQVae(enc, dec, args.emb_dim, args.emb_num)
if args.cuda:
    vqvae.cuda()

optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)

def get_loss(data, vqvae):
    recon_data = vqvae(data)

    # reconst loss
    reconst_loss = F.binary_cross_entropy(recon_data, data)

    # cluster assignment loss
    detach_z_q = Variable(vqvae.z_q.data, requires_grad=False)
    cls_assg_loss = torch.sum((vqvae.z_e - detach_z_q).pow(2))
    cls_assg_loss /= args.batch_size

    # cluster update loss
    detach_z_e = Variable(vqvae.z_e.data, requires_grad=False)
    z_q = vqvae.vqlayer(detach_z_e)
    cls_dist_loss = torch.sum((detach_z_e - z_q).pow(2))
    cls_dist_loss /= args.batch_size

    return reconst_loss, cls_assg_loss, cls_dist_loss

def train(epoch):
    vqvae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        data = data.view(-1, 784)
        if args.cuda:
            data = data.cuda()

        # get losses
        reconst_loss, cls_assg_loss, cls_dist_loss = get_loss(data, vqvae)

        optimizer.zero_grad()
        # get grad for dec and enc
        loss = reconst_loss + args.beta * cls_assg_loss
        loss.backward()

        # clear the grads in vqlayer because they are not grads for updating emb
        vqvae.vqlayer.emb.zero_grad()
        # cluster update loss
        cls_dist_loss.backward() # get grad in emb
        loss += cls_dist_loss

        # all grads good. Update
        optimizer.step()
        train_loss += loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    vqvae.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        data = data.view(-1, 784)

        reconst_loss, cls_assg_loss, cls_dist_loss = get_loss(data, vqvae)
        test_loss += \
            (reconst_loss + args.beta*cls_assg_loss + cls_dist_loss).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

    # sample from each of discrete vector
    samples = vqvae.sample_from_modes()
    save_image(samples.data.view(args.emb_num, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')

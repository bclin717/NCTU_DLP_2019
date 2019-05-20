from __future__ import print_function

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from Lab6.Discriminator import *
from Lab6.Generator import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake', default="mnist")
parser.add_argument('--dataroot', help='path to dataset', default='./')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.MNIST(root=opt.dataroot, download=True,
                     transform=transforms.Compose([
                         transforms.Resize(opt.imageSize),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,)),
                     ]))
nc = 1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")

class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _noise_sample(dis_c, con_c, noise, bs):
    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs), idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)

    return z, idx


losses = {
    'G_loss': [],
    'D_loss': [],
    'Q_loss': []
}

prob = {
    'Real': [],
    'Fake_before Updating G': [],
    'Fake_after Updating G': []
}

def train():

    G = Generator().to(device)
    G.apply(weights_init)

    # Discriminator
    D = Discriminator().to(device)
    D.apply(weights_init)

    # FrontEnd
    FE = FrontEnd().to(device)
    FE.apply(weights_init)

    # Q
    Q = Qnet().to(device)
    Q.apply(weights_init)

    batch_size = 100
    epoch_size = 2

    real_x = torch.FloatTensor(batch_size, 1, 28, 28).cuda()
    label = torch.FloatTensor(batch_size, 1).cuda()
    dis_c = torch.FloatTensor(batch_size, 10).cuda()
    con_c = torch.FloatTensor(batch_size, 2).cuda()
    noise = torch.FloatTensor(batch_size, 62).cuda()

    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    dis_c = Variable(dis_c)
    con_c = Variable(con_c)
    noise = Variable(noise)

    criterionD = nn.BCELoss().cuda()
    criterionQ_dis = nn.CrossEntropyLoss().cuda()
    criterionQ_con = log_gaussian()

    optimD = optim.Adam([{'params': FE.parameters()}, {'params': D.parameters()}], lr=0.00009, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params': G.parameters()}, {'params': Q.parameters()}], lr=1e-3, betas=(0.5, 0.99))

    # fixed random variables
    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    idx = np.arange(10)
    for i in range(9):
        idx = np.append(idx, np.arange(10))

    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1
    fix_noise = torch.Tensor(100, 52).uniform_(-1, 1)

    for epoch in range(epoch_size):
        for num_iters, batch_data in enumerate(dataloader, 0):
            # real part
            optimD.zero_grad()

            x, _ = batch_data

            bs = x.size(0)
            real_x.data.resize_(x.size())
            label.data.resize_(bs)
            dis_c.data.resize_(bs, 10)
            con_c.data.resize_(bs, 2)
            noise.data.resize_(bs, 52)

            real_x.data.copy_(x)
            fe_out1 = FE(real_x)
            probs_real = D(fe_out1)
            label.data.fill_(1)
            loss_real = criterionD(probs_real, label)
            loss_real.backward()

            P_real = probs_real.mean().item()

            # fake part
            z, idx = _noise_sample(dis_c, con_c, noise, bs)
            fake_x = G(z)
            fe_out2 = FE(fake_x.detach())
            probs_fake = D(fe_out2)
            label.data.fill_(0)
            loss_fake = criterionD(probs_fake, label)
            loss_fake.backward()

            P_fake_before = probs_fake.mean().item()


            D_loss = loss_real + loss_fake

            optimD.step()
            # G and Q part
            optimG.zero_grad()

            fe_out = FE(fake_x)
            probs_fake = D(fe_out)
            label.data.fill_(1.0)

            P_fake_after = probs_fake.mean().item()

            reconstruct_loss = criterionD(probs_fake, label)
            q_logits, q_mu, q_var = Q(fe_out)
            class_ = torch.LongTensor(idx).cuda()
            target = Variable(class_)
            dis_loss = criterionQ_dis(q_logits, target)
            con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.1

            G_loss = reconstruct_loss + dis_loss + con_loss
            G_loss.backward()
            optimG.step()

            if num_iters % 100 == 0:
                print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}, Qloss: {4}'.format(
                    epoch, num_iters, D_loss.data.cpu().numpy(),
                    G_loss.data.cpu().numpy(),
                    dis_loss.data.cpu().numpy()))

                losses['G_loss'].append(G_loss.data.cpu().numpy())
                losses['D_loss'].append(D_loss.data.cpu().numpy())
                losses['Q_loss'].append(dis_loss.data.cpu().numpy())

                prob['Real'].append(P_real)
                prob['Fake_before Updating G'].append(P_fake_before)
                prob['Fake_after Updating G'].append(P_fake_after)

                f = open('Loss.txt', 'w+')
                f.write(', '.join(str(e) for e in losses['G_loss']))
                f.write('\n\n')
                f.write(', '.join(str(e) for e in losses['D_loss']))
                f.write('\n\n')
                f.write(', '.join(str(e) for e in losses['Q_loss']))
                f.write('\n\n')
                f.close()

                f = open('Prob.txt', 'w+')
                f.write(', '.join(str(e) for e in prob['Real']))
                f.write('\n\n')
                f.write(', '.join(str(e) for e in prob['Fake_before Updating G']))
                f.write('\n\n')
                f.write(', '.join(str(e) for e in prob['Fake_after Updating G']))
                f.write('\n\n')
                f.close()

                vutils.save_image(x.data, '%s/real.png' % opt.outf, normalize=True, nrow=10)
                noise.data.copy_(fix_noise)
                dis_c.data.copy_(torch.Tensor(one_hot))

                con_c.data.copy_(torch.from_numpy(c1))
                z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
                x_save = G(z)
                vutils.save_image(x_save.data, '%s/model/result_c1_epoch_%02d.png' % (opt.outf, epoch), normalize=True,
                                  nrow=10)

                # con_c.data.copy_(torch.from_numpy(c2))
                # z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
                # x_save = G(z)
                # vutils.save_image(x_save.data, '%s/model/result_c2_epoch_%d.png' % (opt.outf, epoch), normalize=True,
                #                   nrow=10)

        if epoch > 50:
            torch.save(G.state_dict(), '%s/model/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(D.state_dict(), '%s/model/netD_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(FE.state_dict(), '%s/model/netFE_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(Q.state_dict(), '%s/model/netQ_epoch_%d.pth' % (opt.outf, epoch))


def main():
    torch.backends.cudnn.enabled = True
    train()
    print(losses)
    print(prob)
    showResult(title='Loss', results=losses)
    showResult(title='Prob', results=prob)


def showResult(title='', results=losses):
    plt.figure(title, figsize=(15, 7))
    plt.title(title)
    plt.xlabel('Every 100 batch steps')
    plt.ylabel('Loss')
    plt.grid()
    for label, data in results.items():
        plt.plot(range(1, len(data) + 1), data, 'o-' if 'w/o' in label else '-', label=label)
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show()

if __name__ == '__main__':
    main()

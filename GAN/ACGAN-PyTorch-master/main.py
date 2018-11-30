"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from folder import ImageFolder


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
print(opt)

# specify the gpu id if using only 1 gpu
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# datase t
if opt.dataset == 'imagenet':
    # folder dataset
    dataset = ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        classes_idx=(10, 20)
    )
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(
        root=opt.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
else:
    raise NotImplementedError("No such dataset {}".format(opt.dataset))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the generator and initialize the weights
if opt.dataset == 'imagenet':
    netG = _netG(ngpu, nz)
else:
    netG = _netG_CIFAR10(ngpu, nz)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Define the discriminator and initialize the weights
if opt.dataset == 'imagenet':
    netD = _netD(ngpu, num_classes)
else:
    netD = _netD_CIFAR10(ngpu, num_classes)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.batchSize)
aux_label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
eval_label = np.random.randint(0, num_classes, opt.batchSize)
eval_onehot = np.zeros((opt.batchSize, num_classes))
eval_onehot[np.arange(opt.batchSize), eval_label] = 1
eval_noise_[np.arange(opt.batchSize), :num_classes] = eval_onehot[np.arange(opt.batchSize)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.data.resize_as_(real_cpu).copy_(real_cpu)
        dis_label.data.resize_(batch_size).fill_(real_label)
        aux_label.data.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        dis_label.data.fill_(fake_label)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label)  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        errG = dis_errG + aux_errG
        errG.backward()
        D_G_z2 = dis_output.data.mean()
        optimizerG.step()

        # compute the average loss
        curr_iter = epoch * len(dataloader) + i
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter
        all_loss_G += errG.data[0]
        all_loss_D += errD.data[0]
        all_loss_A += accuracy
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], avg_loss_D, errG.data[0], avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf)
            print('Label for eval = {}'.format(eval_label))
            fake = netG(eval_noise)
            print(fake.data.shape)
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
            )

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

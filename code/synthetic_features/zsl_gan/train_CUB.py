import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init

from sklearn.metrics.pairwise import cosine_similarity
import scipy.integrate as integrate
from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import argparse
import os
import random
import glob
import copy
import json
from dataset import FeatDataLayer, LoadDataset
from models import _netD, _netG, _param

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--splitmode', default='easy', type=str, help='the way to split train/test data: easy/hard')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval', type=int, default=40)

opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter for training """
opt.GP_LAMBDA = 10  # Gradient penalty lambda
opt.CENT_LAMBDA = 1
opt.REG_W_LAMBDA = 0.001
opt.REG_Wz_LAMBDA = 0.0001

opt.lr = 0.0001
opt.batchsize = 1000

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20  # knn: the value of K

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

device = 'cpu'


def train():
    param = _param()
    dataset = LoadDataset(opt)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, opt)
    result = Result()
    result_gzsl = Result()
    netG = _netG(dataset.text_dim, dataset.feature_dim).to(device)
    netG.apply(weights_init)
    print(netG)
    netD = _netD(dataset.train_cls_num, dataset.feature_dim).to(device)
    netD.apply(weights_init)
    print(netD)

    exp_info = 'CUB_EASY' if opt.splitmode == 'easy' else 'CUB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(opt.CENT_LAMBDA, opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA)

    out_dir = 'out/{:s}'.format(exp_info)
    out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    for it in range(start_step, 3000 + 1):
        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).to(device)
            X = Variable(torch.from_numpy(feat_data)).to(device)
            y_true = Variable(torch.from_numpy(labels.astype('int'))).to(device)
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).to(device)

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)
            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).to(device)

            X = Variable(torch.from_numpy(feat_data)).to(device)
            y_true = Variable(torch.from_numpy(labels.astype('int'))).to(device)
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).to(device)

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _, C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2

            GC_loss = -G_loss + C_loss

            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0])).to(device)
            if opt.CENT_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        Euclidean_loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                Euclidean_loss *= 1.0 / dataset.train_cls_num * opt.CENT_LAMBDA

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).to(device)
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            # ||W_z||21 regularization, make W_z sparse
            reg_Wz_loss = Variable(torch.Tensor([0.0])).to(device)
            if opt.REG_Wz_LAMBDA != 0:
                Wz = netG.rdc_text.weight
                reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)

            all_loss = GC_loss + Euclidean_loss + reg_loss + reg_Wz_loss
            all_loss.backward()
            optimizerG.step()
            reset_grad(nets)


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()

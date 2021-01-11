'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import random

from models import *
from utils import progress_bar, mixup_data, mixup_criterion, init_params
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='mixup_default', type=str, help='session id')
parser.add_argument('--resume_sess',  type=str, help='resume session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=0.5, type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
base_learning_rate = 0.1

# batch_size = 128
# base_learning_rate = 0.1

if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='/mnt/ramdisk/cifar100/', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='/mnt/Dataset/cifar100/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size * 1.5), shuffle=False, pin_memory=True, num_workers=2)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.resume_sess ,map_location={"cuda" : "cpu"})
    net = shufflenetv2()

    net.cuda()
    net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = PreActResNet18()
    # net = WideResNet_28_10()
    # net = GoogLeNet()
    # net = densenet161()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ResNet50()
    # net = LeNet()
    # net = PNASNetA()
    net = shufflenetv2()
    net.cuda()
    net = torch.nn.DataParallel(net)


result_folder = './results_mwh_100_3/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + '_' + args.sess + '_' + str(args.seed) + '.csv'

# if use_cuda:
print('Using', torch.cuda.device_count(), 'GPUs.')
cudnn.benchmark = True
print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
# criterion = nn.MultiMarginLoss()

# init_params(net)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        mask = random.random()

        if epoch >= 90:
            # threshold = math.cos( math.pi * (epoch - 150) / ((200 - 150) * 2))
            threshold = (100 - epoch) / (100 - 90)
            # threshold = 1.0 - math.cos( math.pi * (200 - epoch) / ((200 - 150) * 2))
            if mask < threshold:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
            else:
                targets_a, targets_b = targets, targets
                lam = 1.0
        elif epoch >= 60:
            if epoch % 2 == 0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
            else:
                targets_a, targets_b = targets, targets
                lam = 1.0
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)

        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = net(inputs)
        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum().item() + (1.0 - lam) * predicted.eq(targets_b.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), (100.*correct)/total, correct, total))
    return (train_loss/batch_idx, 100.*correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.4f%% (%d/%d)'
                % (test_loss/(batch_idx+1), (100.0*correct)/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # print(acc)
    if acc > best_acc:
        best_acc = acc
    if (epoch + 1) % 50 == 0 :
        checkpoint(net, acc, epoch)
    return (test_loss/batch_idx, 100.*correct/total)

def checkpoint(net, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }
    if not os.path.isdir('./checkpoint/'):
        os.makedirs('./checkpoint/')
    torch.save(state, './checkpoint/ckpt.t7.' + args.sess + '_' + str(epoch))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 50 :
        lr /= 10
    if epoch >= 75 :
        lr /= 10

    # if epoch <= 9 and lr > 0.1:
    #     # warm-up training for large minibatch
    #     lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    # if epoch >= 60 * rate :
    #     lr /= 10
    # if epoch >= 120 * rate:
    #     lr /= 10
    # if epoch >= 180 * rate:
    #     lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, int(100)):
    adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
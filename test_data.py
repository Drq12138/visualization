import argparse
from enum import EnumMeta
import os
from statistics import mode

from numpy import save
from utils import get_model, set_seed, get_datasets, AverageMeter, accuracy
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def train_net(model, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = AverageMeter()

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))
    
    return losses.avg

def test(model, test_loader, criterion):
    model.eval()
    accuracys = AverageMeter()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            acc = accuracy(output.data, target)[0]
            accuracys.update(acc.item(),data.size(0))
            pass
    
    return accuracys.avg



def main():
    parser = argparse.ArgumentParser(description='a single test to get familiar with the code')
    parser.add_argument('--arch', default='resnet32')
    parser.add_argument('--datasets', default='CIFAR10')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--smalldatasets', default=None, type=float)
    parser.add_argument('--mult_gpu', action='store_true')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--save_dir', default='./../checkpoints/visualization')
    parser.add_argument('--name', default='test_visualization')



    args = parser.parse_args()
    set_seed(args.randomseed)

    train_loader, val_loader = get_datasets(args)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    model = get_model(args)
    if args.mult_gpu:
        model = torch.nn.DataParallel(model)

    model.cuda()
    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    
    if args.datasets == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
                                                            
    elif args.datasets == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150])
    
    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
    
    save_path = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    final_checkpoint = os.path.join(save_path, 'save_net_' + args.arch + '_' + str(args.epoch-1))
    if os.path.isfile(final_checkpoint):
        print('you have trained before ...')
    else:
        torch.save(model.state_dict(), os.path.join(save_path, 'save_net_' + args.arch + '_' + str(0)))
        orig_loss = []
        orig_acc = []
        for epoch in range(args.epoch):
            print("Epoch %i"%(epoch))
            tloss = train_net(model, train_loader, optimizer, criterion, epoch)
            orig_loss.append(tloss)
            accu = test(model, val_loader, criterion)
            orig_acc.append(accu)
            torch.save(model.state_dict(), os.path.join(save_path, 'save_net_' + args.arch + '_' + str(epoch)))

        np.save(os.path.join(save_path, 'save_net_' + args.arch + '_orig_loss' ), orig_loss)
        np.save(os.path.join(save_path, 'save_net_' + args.arch + '_orig_acc' ), orig_acc)











    


    pass

if __name__ == "__main__":
    main()
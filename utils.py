import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import resnet


def set_seed(seed=233): 
    print ('Random Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(args):
    if args.datasets == 'MNIST':
        print ('normal dataset!')
        mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
        mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 100, shuffle=False)
    elif args.datasets == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.smalldatasets:
            percent = args.smalldatasets
            path = os.path.join("../data", 'cifar10_' + str(percent) +  '_smalldataset')
            # path = 'cifar10_' + str(percent) +  '_smalldataset'
            print ('Use ', percent, 'of Datasets')
            print ('path:', path)
            ################################################################
            # Use small datasets

            if os.path.exists(path):
                print ('read dataset!')
                trainset = torch.load(path)
            else:
                print ('make dataset!')
                trainset = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,
                    ]), download=True)
                N = int(percent * len(trainset))
                trainset.targets = trainset.targets[:N]
                trainset.data = trainset.data[:N]

                torch.save(trainset, path)
                print (N)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            print ('dataset size: ', len(train_loader.dataset))
        else:
            print ('normal dataset!')
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=True),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    elif args.datasets == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        if args.smalldatasets:
            percent = args.smalldatasets
            path = os.path.join("../data", 'cifar100_' + str(percent) +  '_smalldataset')
            # path = 'cifar100_' + str(percent) +  '_smalldataset'
            print ('Use ', percent, 'of Datasets')
            print ('path:', path)
            ################################################################
            # Use small datasets

            if os.path.exists(path):
                print ('load dataset!')
                trainset = torch.load(path)
            else:
                print ('create dataset!')
                trainset = datasets.CIFAR100(root='../data', train=True, transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,
                    ]), download=True)
                N = int(percent * len(trainset))
                trainset.targets = trainset.targets[:N]
                trainset.data = trainset.data[:N]

                torch.save(trainset, path)
                print (N)
                

            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            print ('dataset size: ', len(train_loader.dataset))
        
        else:
            print ('normal dataset!')
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root='../data', train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=True),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader


def get_model(args):
    if args.datasets == 'CIFAR10' or args.datasets == 'MNIST':
        num_class = 10
    elif args.datasets == 'CIFAR100':
        num_class = 100

    if args.datasets == 'CIFAR100':
        if args.arch == 'vgg16':
            from models.vgg import vgg16_bn
            net = vgg16_bn()
        elif args.arch == 'vgg13':
            from models.vgg import vgg13_bn
            net = vgg13_bn()
        elif args.arch == 'vgg11':
            from models.vgg import vgg11_bn
            net = vgg11_bn()
        elif args.arch == 'vgg19':
            from models.vgg import vgg19_bn
            net = vgg19_bn()
        elif args.arch == 'densenet121':
            from models.densenet import densenet121
            net = densenet121()
        elif args.arch == 'densenet161':
            from models.densenet import densenet161
            net = densenet161()
        elif args.arch == 'densenet169':
            from models.densenet import densenet169
            net = densenet169()
        elif args.arch == 'densenet201':
            from models.densenet import densenet201
            net = densenet201()
        elif args.arch == 'googlenet':
            from models.googlenet import googlenet
            net = googlenet()
        elif args.arch == 'inceptionv3':
            from models.inceptionv3 import inceptionv3
            net = inceptionv3()
        elif args.arch == 'inceptionv4':
            from models.inceptionv4 import inceptionv4
            net = inceptionv4()
        elif args.arch == 'inceptionresnetv2':
            from models.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2()
        elif args.arch == 'xception':
            from models.xception import xception
            net = xception()
        elif args.arch == 'resnet18':
            from resnet import resnet18
            net = resnet18()
        elif args.arch == 'resnet34':
            from resnet import resnet34
            net = resnet34()
        elif args.arch == 'resnet50':
            from resnet import resnet50
            net = resnet50()
        elif args.arch == 'resnet101':
            from resnet import resnet101
            net = resnet101()
        elif args.arch == 'resnet152':
            from resnet import resnet152
            net = resnet152()
        elif args.arch == 'preactresnet18':
            from models.preactresnet import preactresnet18
            net = preactresnet18()
        elif args.arch == 'preactresnet34':
            from models.preactresnet import preactresnet34
            net = preactresnet34()
        elif args.arch == 'preactresnet50':
            from models.preactresnet import preactresnet50
            net = preactresnet50()
        elif args.arch == 'preactresnet101':
            from models.preactresnet import preactresnet101
            net = preactresnet101()
        elif args.arch == 'preactresnet152':
            from models.preactresnet import preactresnet152
            net = preactresnet152()
        elif args.arch == 'resnext50':
            from models.resnext import resnext50
            net = resnext50()
        elif args.arch == 'resnext101':
            from models.resnext import resnext101
            net = resnext101()
        elif args.arch == 'resnext152':
            from models.resnext import resnext152
            net = resnext152()
        elif args.arch == 'shufflenet':
            from models.shufflenet import shufflenet
            net = shufflenet()
        elif args.arch == 'shufflenetv2':
            from models.shufflenetv2 import shufflenetv2
            net = shufflenetv2()
        elif args.arch == 'squeezenet':
            from models.squeezenet import squeezenet
            net = squeezenet()
        elif args.arch == 'mobilenet':
            from models.mobilenet import mobilenet
            net = mobilenet()
        elif args.arch == 'mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        elif args.arch == 'nasnet':
            from models.nasnet import nasnet
            net = nasnet()
        elif args.arch == 'attention56':
            from models.attention import attention56
            net = attention56()
        elif args.arch == 'attention92':
            from models.attention import attention92
            net = attention92()
        elif args.arch == 'seresnet18':
            from models.senet import seresnet18
            net = seresnet18()
        elif args.arch == 'seresnet34':
            from models.senet import seresnet34
            net = seresnet34()
        elif args.arch == 'seresnet50':
            from models.senet import seresnet50
            net = seresnet50()
        elif args.arch == 'seresnet101':
            from models.senet import seresnet101
            net = seresnet101()
        elif args.arch == 'seresnet152':
            from models.senet import seresnet152
            net = seresnet152()
        elif args.arch == 'wideresnet':
            from models.wideresidual import wideresnet
            net = wideresnet()
        elif args.arch == 'stochasticdepth18':
            from models.stochasticdepth import stochastic_depth_resnet18
            net = stochastic_depth_resnet18()
        elif args.arch == 'efficientnet':
            from models.efficientnet import efficientnet
            net = efficientnet(1, 1, 100, bn_momentum=0.9)
        elif args.arch == 'stochasticdepth34':
            from models.stochasticdepth import stochastic_depth_resnet34
            net = stochastic_depth_resnet34()
        elif args.arch == 'stochasticdepth50':
            from models.stochasticdepth import stochastic_depth_resnet50
            net = stochastic_depth_resnet50()
        elif args.arch == 'stochasticdepth101':
            from models.stochasticdepth import stochastic_depth_resnet101
            net = stochastic_depth_resnet101()
        else:
            net = resnet.__dict__[args.arch](num_classes=num_class)

        return net
    return resnet.__dict__[args.arch](num_classes=num_class)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
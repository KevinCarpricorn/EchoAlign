import numpy as np
import torch
import torchvision.transforms as transforms


def transform(args):
    if args.dataset.startswith('cifar100'):
        args.num_classes = 100
        args.model = 'resnet34'
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    elif args.dataset.startswith('cifar10'):
        args.num_classes = 10
        args.model = 'resnet18'
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target

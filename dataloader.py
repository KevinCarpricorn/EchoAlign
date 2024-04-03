from torch.utils.data import DataLoader

import dataset


def get_noisy_dataset(train, transform, target_transform, args, exist=False):
    if args.dataset == 'cifar10':
        data = dataset.CIFAR10(args, train=train, transform=transform, target_transform=target_transform, exist=exist)
    elif args.dataset == 'cifar100':
        data = dataset.CIFAR100(args, train=train, transform=transform, target_transform=target_transform, exist=exist)
    elif args.dataset == 'clothing1m':
        pass

    return data


def get_processed_dataset(train, transform, target_transform, args, exist=False):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data = dataset.processed_dataset(args, train=train, transform=transform, target_transform=target_transform,
                                         exist=exist)
    elif args.dataset == 'clothing1m':
        pass

    return data


def get_distilled_dataset(train, transform, target_transform, dir, args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data = dataset.distilled_dataset(train=train, transform=transform, target_transform=target_transform, dir=dir)
    elif args.dataset == 'clothing1m':
        pass

    return data


def get_test_loader(transform, target_transform, args, exist=False):
    if args.dataset == 'cifar10':
        test_data = dataset.CIFAR10_test(transform=transform, target_transform=target_transform, exist=exist)
    elif args.dataset == 'cifar100':
        test_data = dataset.CIFAR100_test(transform=transform, target_transform=target_transform, exist=exist)
    elif args.dataset == 'clothing1m':
        pass

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             drop_last=False)

    return test_loader

import numpy as np
import torch.utils.data as Data
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import utils
import torch


class CIFAR10(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1,
                 num_class=10):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        dir = './data/cifar10'
        os.makedirs(dir, exist_ok=True)

        # check existance of train dataset
        if os.path.exists(os.path.join(dir, 'train_images.npy')):
            self.train_image = np.load(os.path.join(dir, 'train_images.npy'))
            self.train_label = np.load(os.path.join(dir, 'train_labels.npy'))
            self.val_image = np.load(os.path.join(dir, 'val_images.npy'))
            self.val_label = np.load(os.path.join(dir, 'val_labels.npy'))
            self.clean_train_label = np.load(os.path.join(dir, 'clean_train_labels.npy'))
            self.clean_val_label = np.load(os.path.join(dir, 'clean_val_labels.npy'))
        else:
            # download dataset
            train_set = datasets.CIFAR10(root=dir, train=True, download=True, transform=transforms.ToTensor())
            test_set = datasets.CIFAR10(root=dir, train=False, download=True, transform=self.test_transform)

            # split dataset
            train_image = train_set.data
            train_label = np.array(train_set.targets)
            test_image = test_set.data
            test_label = np.array(test_set.targets)

            # save test dataset
            np.save(os.path.join(dir, 'test_images.npy'), test_image)
            np.save(os.path.join(dir, 'test_labels.npy'), test_label)

            # split train dataset
            self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.dataset_split(
                train_image,
                train_label,
                noise_rate,
                split_per,
                random_seed,
                num_class)
            np.save(os.path.join(dir, 'train_images.npy'), self.train_image)
            np.save(os.path.join(dir, 'train_labels.npy'), self.train_label)
            np.save(os.path.join(dir, 'val_images.npy'), self.val_image)
            np.save(os.path.join(dir, 'val_labels.npy'), self.val_label)
            np.save(os.path.join(dir, 'clean_train_labels.npy'), self.clean_train_label)
            np.save(os.path.join(dir, 'clean_val_labels.npy'), self.clean_val_label)

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_image[index], self.train_label[index]
        else:
            img, label = self.val_image[index], self.val_label[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_image) if self.train else len(self.val_image)


class processed_CIFAR10(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        processed_dir = './data/cifar10/processed'
        label_dir = './data/cifar10'

        # check existance of train dataset
        self.train_image = np.load(os.path.join(processed_dir, 'processed_train_images.npy'))
        self.train_image = self.resize_images(self.train_image)
        self.train_label = np.load(os.path.join(label_dir, 'train_labels.npy'))
        self.val_image = np.load(os.path.join(processed_dir, 'processed_val_images.npy'))
        self.val_image = self.resize_images(self.val_image)
        self.val_label = np.load(os.path.join(label_dir, 'val_labels.npy'))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_image[index], self.train_label[index]
        else:
            img, label = self.val_image[index], self.val_label[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_image) if self.train else len(self.val_image)

    def resize_images(self, images, size=(32, 32)):
        resized_images = np.empty((images.shape[0], size[0], size[1], images.shape[3]), dtype=np.uint8)
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i].astype('uint8'))
            img_resized = img.resize(size)
            resized_images[i] = np.array(img_resized)
        return resized_images


class distilled_CIFAR10(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dir=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if dir is None:
            distilled_dataset_dir = './data/cifar10/distilled_dataset'
            self.train_image = np.load(os.path.join(distilled_dataset_dir, 'distilled_train_images.npy'))
            self.train_label = np.load(os.path.join(distilled_dataset_dir, 'distilled_train_labels.npy'))
            self.val_image = np.load(os.path.join(distilled_dataset_dir, 'distilled_val_images.npy'))
            self.val_label = np.load(os.path.join(distilled_dataset_dir, 'distilled_val_labels.npy'))
        else:
            distilled_dataset_dir = dir
            self.val_image = np.load(os.path.join(distilled_dataset_dir, 'distilled_val_images.npy'))
            self.val_label = np.load(os.path.join(distilled_dataset_dir, 'distilled_val_labels.npy'))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_image[index], self.train_label[index]
        else:
            img, label = self.val_image[index], self.val_label[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_image) if self.train else len(self.val_image)


class source_CIFAR10(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        dataset_dir = './data/cifar10/source_target_dataset'
        self.source_image = np.load(os.path.join(dataset_dir, 'source_images.npy'))
        self.source_label = np.load(os.path.join(dataset_dir, 'source_labels.npy'))
        self.classes = np.load(os.path.join(dataset_dir, 'classes.npy'))

    def __getitem__(self, index):
        img, label, cls = self.source_image[index], self.source_label[index], self.classes[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, cls, index

    def __len__(self):
        return len(self.source_image)


class target_CIFAR10(Data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        dataset_dir = './data/cifar10/source_target_dataset'
        self.target_image = np.load(os.path.join(dataset_dir, 'target_images.npy'))

    def __getitem__(self, index):
        img = self.target_image[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.target_image)


class CIFAR10_test(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.test_image = np.load('./data/cifar10/test_images.npy')
        self.test_label = np.load('./data/cifar10/test_labels.npy')

    def __getitem__(self, index):
        img, label = self.test_image[index], self.test_label[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_image)


class CustomDataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)

    def get_data(self, num):
        data = []
        for _ in range(num):
            try:
                data.append(next(self.iterator))
            except StopIteration:
                self.iterator = iter(self.data_loader)
                data.append(next(self.iterator))
        data = torch.cat(data, dim=0)
        return data
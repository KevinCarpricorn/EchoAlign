import numpy as np
import torch.utils.data as Data
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import utils


class CIFAR10(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1,
                 num_class=10):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        dir = './data/cifar10'
        os.makedirs(dir, exist_ok=True)

        # check existance of train dataset
        if os.path.exists(os.path.join(dir, 'train_images.npy')):
            self.train_image = np.load(os.path.join(dir, 'train_images.npy'))
            self.train_label = np.load(os.path.join(dir, 'train_labels.npy'))
            self.val_image = np.load(os.path.join(dir, 'val_images.npy'))
            self.val_label = np.load(os.path.join(dir, 'val_labels.npy'))
        else:
            # download dataset
            train_set = datasets.CIFAR10(root=dir, train=True, download=True, transform=transforms.ToTensor())
            test_set = datasets.CIFAR10(root=dir, train=False, download=True, transform=transforms.ToTensor())

            # split dataset
            train_image = np.array([np.array(sample[0]) for sample in train_set])
            train_label = np.array([np.array(sample[1]) for sample in train_set])
            test_image = np.array([np.array(sample[0]) for sample in test_set])
            test_label = np.array([np.array(sample[1]) for sample in test_set])

            # save test dataset
            np.save(os.path.join(dir, 'test_images.npy'), test_image)
            np.save(os.path.join(dir, 'test_labels.npy'), test_label)

            # split train dataset
            self.train_image, self.train_label, self.val_image, self.val_label = utils.dataset_split(train_image,
                                                                                                     train_label,
                                                                                                     noise_rate,
                                                                                                     split_per,
                                                                                                     random_seed,
                                                                                                     num_class)
            np.save(os.path.join(dir, 'train_images.npy'), self.train_image)
            np.save(os.path.join(dir, 'train_labels.npy'), self.train_label)
            np.save(os.path.join(dir, 'val_images.npy'), self.val_image)
            np.save(os.path.join(dir, 'val_labels.npy'), self.val_label)

        if self.train:
            self.train_image = self.train_image.reshape((45000, 3, 32, 32))
            self.train_image = self.train_image.transpose((0, 2, 3, 1))
        else:
            self.val_image = self.val_image.reshape((5000, 3, 32, 32))
            self.val_image = self.val_image.transpose((0, 2, 3, 1))

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

        return img, label

    def __len__(self):
        return len(self.train_image) if self.train else len(self.val_image)


class CIFAR10_test(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.test_image = np.load('./data/cifar10/test_images.npy')
        self.test_label = np.load('./data/cifar10/test_labels.npy')
        self.test_image = self.test_image.reshape((10000, 3, 32, 32))
        self.test_image = self.test_image.transpose((0, 2, 3, 1))

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
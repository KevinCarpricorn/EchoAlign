import os
import pickle
import sys

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

import utils


class CIFAR10_for_instance(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10_for_instance, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.train = train  # training set or test set
        self.do_data_augmentation = set()
        self.to_tensor = transforms.ToTensor()

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # 如果你准备做augmentation ratio 实验，启用下面这行代码
        #if self.transform is not None and index in self.do_data_augmentation:
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.to_tensor(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR10(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1,
                 num_class=10, noise_type='symmetric'):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        if noise_type == 'symmetric':
            dir = f'./data/cifar10/symmetric_noise_{noise_rate}'
        elif noise_type == 'instance':
            dir = f'./data/cifar10/instance_noise_{noise_rate}'
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
            test_set = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=self.test_transform)

            # split dataset
            test_image = test_set.data
            test_label = np.array(test_set.targets)

            # save test dataset
            np.save(os.path.join('./data/cifar10', 'test_images.npy'), test_image)
            np.save(os.path.join('./data/cifar10', 'test_labels.npy'), test_label)

            # split train dataset
            if noise_type == 'symmetric':
                train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                                             transform=transforms.ToTensor())
                train_image = train_set.data
                train_label = np.array(train_set.targets)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.symmetric_dataset_split(
                    train_image,
                    train_label,
                    noise_rate,
                    split_per,
                    random_seed,
                    num_class)
            elif noise_type == 'instance':
                train_set = CIFAR10_for_instance(root=dir, train=True, transform=transforms.ToTensor(), download=True)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.instance_dataset_split(
                    train_set,
                    train_set.targets,
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
    def __init__(self, train=True, transform=None, target_transform=None, noise_type='symmetric'):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        processed_dir = './data/cifar10/processed'
        if noise_type == 'symmetric':
            label_dir = './data/cifar10/symmetric_noise_0.5'
            processed_dir = os.path.join(processed_dir, 'symmetric_noise_0.5')
        elif noise_type == 'instance':
            label_dir = './data/cifar10/instance_noise_0.5'
            processed_dir = os.path.join(processed_dir, 'instance_noise_0.5')

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
import os
import pickle
import sys
import tarfile

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import VisionDataset, ImageFolder
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
        # if self.transform is not None and index in self.do_data_augmentation:
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


class CIFAR100_for_instance(CIFAR10_for_instance):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class CIFAR10(Data.Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None, split_per=0.9, exist=False):
        self.transform = transform
        self.target_transform = target_transform
        self.noise_rate = args.noise_rate
        self.random_seed = args.seed
        self.noise_type = args.noise_type
        self.num_classes = args.num_classes
        self.exist = exist
        self.train = train

        train_dir = f'./data/cifar10/{self.noise_type}_{self.noise_rate}/train'
        val_dir = f'./data/cifar10/{self.noise_type}_{self.noise_rate}/val'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # check existence of the dataset
        if self.exist:
            if self.train:
                self.train_image = np.load(os.path.join(train_dir, 'train_images.npy'))
                self.train_label = np.load(os.path.join(train_dir, 'train_labels.npy'))
                self.clean_train_label = np.load(os.path.join(train_dir, 'clean_train_labels.npy'))
            else:
                self.val_image = np.load(os.path.join(val_dir, 'val_images.npy'))
                self.val_label = np.load(os.path.join(val_dir, 'val_labels.npy'))
                self.clean_val_label = np.load(os.path.join(val_dir, 'clean_val_labels.npy'))
        else:
            # split train dataset
            if self.noise_type == 'symmetric':
                train_set = datasets.CIFAR10(root='./data/cifar10/base', train=True, download=True,
                                             transform=transforms.ToTensor())
                train_image = train_set.data
                train_label = np.array(train_set.targets)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.symmetric_dataset_split(
                    train_image,
                    train_label,
                    self.noise_rate,
                    split_per,
                    self.random_seed,
                    self.num_classes)
            elif self.noise_type == 'instance':
                train_set = CIFAR10_for_instance(root=f'./data/cifar10/base', train=True,
                                                 transform=transforms.ToTensor(), download=True)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.instance_dataset_split(
                    train_set,
                    train_set.targets,
                    self.noise_rate,
                    split_per,
                    self.random_seed,
                    args)
            else:
                train_set = datasets.CIFAR10(root='./data/cifar10/base', train=True, download=True,
                                             transform=transforms.ToTensor())
                train_image = train_set.data
                train_label = np.array(train_set.targets)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.pairflip_dataset_split(
                    train_image,
                    train_label,
                    self.noise_rate,
                    split_per,
                    self.random_seed,
                    self.num_classes)

            np.save(os.path.join(train_dir, 'train_images.npy'), self.train_image)
            np.save(os.path.join(train_dir, 'train_labels.npy'), self.train_label)
            np.save(os.path.join(train_dir, 'clean_train_labels.npy'), self.clean_train_label)
            np.save(os.path.join(val_dir, 'val_images.npy'), self.val_image)
            np.save(os.path.join(val_dir, 'val_labels.npy'), self.val_label)
            np.save(os.path.join(val_dir, 'clean_val_labels.npy'), self.clean_val_label)

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


class CIFAR100(Data.Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None, split_per=0.9, exist=False):
        self.transform = transform
        self.target_transform = target_transform
        self.noise_rate = args.noise_rate
        self.random_seed = args.seed
        self.noise_type = args.noise_type
        self.num_classes = args.num_classes
        self.exist = exist
        self.train = train

        train_dir = f'./data/cifar100/{self.noise_type}_{self.noise_rate}/train'
        val_dir = f'./data/cifar100/{self.noise_type}_{self.noise_rate}/val'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # check existence of the dataset
        if self.exist:
            if self.train:
                self.train_image = np.load(os.path.join(train_dir, 'train_images.npy'))
                self.train_label = np.load(os.path.join(train_dir, 'train_labels.npy'))
                self.clean_train_label = np.load(os.path.join(train_dir, 'clean_train_labels.npy'))
            else:
                self.val_image = np.load(os.path.join(val_dir, 'val_images.npy'))
                self.val_label = np.load(os.path.join(val_dir, 'val_labels.npy'))
                self.clean_val_label = np.load(os.path.join(val_dir, 'clean_val_labels.npy'))
        else:
            # split train dataset
            if self.noise_type == 'symmetric':
                train_set = datasets.CIFAR100(root='./data/cifar100/base', train=True, download=True,
                                              transform=transforms.ToTensor())
                train_image = train_set.data
                train_label = np.array(train_set.targets)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.symmetric_dataset_split(
                    train_image,
                    train_label,
                    self.noise_rate,
                    split_per,
                    self.random_seed,
                    self.num_classes)
            elif self.noise_type == 'instance':
                train_set = CIFAR100_for_instance(root=f'./data/cifar100/base', train=True,
                                                  transform=transforms.ToTensor(), download=True)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.instance_dataset_split(
                    train_set,
                    train_set.targets,
                    self.noise_rate,
                    split_per,
                    self.random_seed,
                    args)
            else:
                train_set = datasets.CIFAR100(root='./data/cifar100/base', train=True, download=True,
                                              transform=transforms.ToTensor())
                train_image = train_set.data
                train_label = np.array(train_set.targets)
                self.train_image, self.train_label, self.val_image, self.val_label, self.clean_train_label, self.clean_val_label = utils.pairflip_dataset_split(
                    train_image,
                    train_label,
                    self.noise_rate,
                    split_per,
                    self.random_seed,
                    self.num_classes)

            np.save(os.path.join(train_dir, 'train_images.npy'), self.train_image)
            np.save(os.path.join(train_dir, 'train_labels.npy'), self.train_label)
            np.save(os.path.join(train_dir, 'clean_train_labels.npy'), self.clean_train_label)
            np.save(os.path.join(val_dir, 'val_images.npy'), self.val_image)
            np.save(os.path.join(val_dir, 'val_labels.npy'), self.val_label)
            np.save(os.path.join(val_dir, 'clean_val_labels.npy'), self.clean_val_label)

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


class Clothing1M_Dataset(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        data_dir = os.getenv('PBS_JOBFS')
        self.train = train
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.target_transform = target_transform

        if self.train:
            data_path = os.path.join(data_dir, 'noisy')
            super().__init__(data_path, transform=self.transform, target_transform=self.target_transform)
        else:
            data_path = os.path.join(data_dir, 'clean_val')
            super().__init__(data_path, transform=self.transform, target_transform=self.target_transform)

    def __getitem__(self, index):
        image, label = super(Clothing1M_Dataset, self).__getitem__(index)

        return image, label, index


class Clothing1M_processed(ImageFolder):
    def __init__(self, transform=None, target_transform=None):
        data_dir = os.getenv('PBS_JOBFS')

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.target_transform = target_transform

        data_path = os.path.join(data_dir, 'processed_noisy')
        super().__init__(data_path, transform=self.transform, target_transform=self.target_transform)

    def __getitem__(self, index):
        image, label = super(Clothing1M_processed, self).__getitem__(index)

        return image, label, index


class processed_dataset(Data.Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None, exist=True):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.exist = exist
        self.noise_type = args.noise_type
        self.noise_rate = args.noise_rate
        self.dataset = args.dataset

        train_dir = f'./data/{self.dataset}/{self.noise_type}_{self.noise_rate}/train'
        val_dir = f'./data/{self.dataset}/{self.noise_type}_{self.noise_rate}/val'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        if self.exist:
            if self.train:
                self.train_image = np.load(os.path.join(train_dir, 'processed_train_images.npy'))
                self.train_image = self.resize_images(self.train_image)
                self.train_label = np.load(os.path.join(train_dir, 'train_labels.npy'))
            else:
                self.val_image = np.load(os.path.join(val_dir, 'processed_val_images.npy'))
                self.val_image = self.resize_images(self.val_image)
                self.val_label = np.load(os.path.join(val_dir, 'val_labels.npy'))
        else:
            # raise NotProcessedDatasetError
            raise Exception('NotProcessedDatasetError')

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


class distilled_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dir=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dir = dir

        if train:
            distilled_dataset_dir = os.path.join(dir, 'train')
            self.train_image = np.load(os.path.join(distilled_dataset_dir, 'distilled_train_images.npy'))
            self.train_label = np.load(os.path.join(distilled_dataset_dir, 'distilled_train_labels.npy'))
            self.classes = np.load(os.path.join(distilled_dataset_dir, 'classes.npy'))
        else:
            distilled_dataset_dir = os.path.join(dir, 'val')
            self.val_image = np.load(os.path.join(distilled_dataset_dir, 'distilled_val_images.npy'))
            self.val_label = np.load(os.path.join(distilled_dataset_dir, 'distilled_val_labels.npy'))

    def __getitem__(self, index):
        if self.train:
            img, label, cls = self.train_image[index], self.train_label[index], self.classes[index]
        else:
            img, label = self.val_image[index], self.val_label[index]
            cls = 0

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, cls, index

    def __len__(self):
        return len(self.train_image) if self.train else len(self.val_image)


class distilled_dataset_Clothing1M(Data.Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None, dir=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dir = dir
        self.args = args
        data_dir = os.getenv('PBS_JOBFS')

        if train:
            self.processed_train_data = Clothing1M_processed(transform=transform,
                                                             target_transform=target_transform)
            distilled_clean_dataset_dir = os.path.join(dir, 'distilled_clean_images.tar')
            with tarfile.open(distilled_clean_dataset_dir, 'r') as tar:
                tar.extractall(path=data_dir)
            self.distilled_clean_data = datasets.ImageFolder(os.path.join(data_dir, 'distilled_clean_images'),
                                                             transform=transform, target_transform=target_transform)
        else:
            distilled_dataset_dir = os.path.join(data_dir, 'clean_val')
            self.distilled_val_data = datasets.ImageFolder(distilled_dataset_dir, transform=transform,
                                                           target_transform=target_transform)

    def __getitem__(self, index):
        if self.train:
            if index < len(self.processed_train_data):
                img, label, index = self.processed_train_data[index]
                cls = 1
            else:
                img, label = self.distilled_clean_data[index - len(self.processed_train_data)]
                cls = 0
        else:
            img, label = self.distilled_val_data[index]
            cls = 0

        return img, label, cls, index

    def __len__(self):
        return len(self.processed_train_data) + len(self.distilled_clean_data) if self.train else len(
            self.distilled_val_data)


class CIFAR10_test(Data.Dataset):
    def __init__(self, transform=None, target_transform=None, exist=False):
        self.transform = transform
        self.target_transform = target_transform
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.exist = exist

        dir = './data/cifar10/test'
        os.makedirs(dir, exist_ok=True)
        if self.exist:
            self.test_image = np.load(os.path.join(dir, 'test_images.npy'))
            self.test_label = np.load(os.path.join(dir, 'test_labels.npy'))
        else:
            test_set = datasets.CIFAR10(root='./data/cifar10/base', train=False, download=True,
                                        transform=self.test_transform)

            self.test_image = test_set.data
            self.test_label = np.array(test_set.targets)

            np.save(os.path.join(dir, 'test_images.npy'), self.test_image)
            np.save(os.path.join(dir, 'test_labels.npy'), self.test_label)

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


class CIFAR100_test(Data.Dataset):
    def __init__(self, transform=None, target_transform=None, exist=False):
        self.transform = transform
        self.target_transform = target_transform
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
        self.exist = exist

        dir = './data/cifar100/test'
        os.makedirs(dir, exist_ok=True)
        if self.exist:
            self.test_image = np.load(os.path.join(dir, 'test_images.npy'))
            self.test_label = np.load(os.path.join(dir, 'test_labels.npy'))
        else:
            test_set = datasets.CIFAR100(root='./data/cifar100/base', train=False, download=True,
                                         transform=self.test_transform)

            self.test_image = test_set.data
            self.test_label = np.array(test_set.targets)

            np.save(os.path.join(dir, 'test_images.npy'), self.test_image)
            np.save(os.path.join(dir, 'test_labels.npy'), self.test_label)

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


class Clothing1M_test(ImageFolder):
    def __init__(self, transform=None, target_transform=None):
        data_dir = os.getenv('PBS_JOBFS')

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.target_transform = target_transform

        data_path = os.path.join(data_dir, 'clean_test')
        super().__init__(data_path, transform=self.transform, target_transform=self.target_transform)


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
        return data[0] if len(data) == 1 else torch.cat(data, dim=0)


class filtered_dataset(Data.Dataset):
    def __init__(self, train, transform=None, target_transform=None, dir=None):
        self.transform = transform
        self.target_transform = target_transform
        self.dir = dir

        if train:
            dataset_dir = os.path.join(dir, 'train')
        else:
            dataset_dir = os.path.join(dir, 'val')
        self.fine_tune_image = np.load(os.path.join(dataset_dir, 'distilled_clean_images.npy'))
        self.fine_tune_label = np.load(os.path.join(dataset_dir, 'distilled_clean_labels.npy'))

    def __getitem__(self, index):
        img, label = self.fine_tune_image[index], self.fine_tune_label[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.fine_tune_image)


class filtered_dataset_Clothing1M(Data.Dataset):
    def __init__(self, train, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        data_dir = os.getenv('PBS_JOBFS')

        if train:
            self.distilled_clean_data = datasets.ImageFolder(os.path.join(data_dir, 'distilled_clean_images'),
                                                             transform=transform, target_transform=target_transform)
        else:
            distilled_dataset_dir = os.path.join(data_dir, 'clean_val')
            self.distilled_val_data = datasets.ImageFolder(distilled_dataset_dir, transform=transform,
                                                           target_transform=target_transform)

    def __getitem__(self, index):
        if self.train:
            img, label = self.distilled_clean_data[index]
        else:
            img, label = self.distilled_val_data[index]

        return img, label, index

    def __len__(self):
        return len(self.distilled_clean_data) if self.train else len(self.distilled_val_data)

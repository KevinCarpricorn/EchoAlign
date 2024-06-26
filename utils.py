import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from scipy import stats


def multiclass_noisify(train_labels, P, random_state=1):
    assert P.shape[0] == P.shape[1]
    assert np.max(train_labels) < P.shape[0]

    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    new_labels = train_labels.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(train_labels.shape[0]):
        i = train_labels[idx]
        if not isinstance(i, np.ndarray):
            i = [i]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_labels[idx] = np.where(flipped == 1)[0]

    return new_labels


def v6_get_noisy_label(n, dataset, labels, args=None):
    if args.dataset == 'cifar10':
        label_num = 10
    elif args.dataset == 'cifar100':
        label_num = 100
    norm_std = 0.1
    os.makedirs(f'./data/{args.dataset}/{args.noise_type}_{args.noise_rate}', exist_ok=True)
    file_path = f'./data/{args.dataset}/{args.noise_type}_{args.noise_rate}/v6_{args.dataset}_labels_{args.noise_rate}.npy'
    if os.path.exists(file_path):
        new_label = np.load(file_path)
    else:
        from math import inf
        P = []
        if args.dataset == 'mnist':
            feature_size = 28 * 28
        else:
            feature_size = 3 * 32 * 32

        flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n,
                                            scale=norm_std)
        flip_rate = flip_distribution.rvs(dataset.data.shape[0])

        if not isinstance(labels, torch.FloatTensor):
            try:
                labels = torch.FloatTensor(labels)
            except TypeError:
                labels = labels.float()

        labels = labels

        W = np.random.randn(label_num, feature_size, label_num)
        W = torch.FloatTensor(W)
        for i, (x, y, idx) in enumerate(dataset):
            # 1*m *  m*10 = 1*10
            x = x
            # print(x.type(), end='\n\n\n')
            A = x.view(1, -1).mm(W[y]).squeeze(0)
            A[y] = -inf
            A = flip_rate[i] * torch.softmax(A, dim=0)
            A[y] += 1 - flip_rate[i]
            P.append(A)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(label_num)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(len(dataset))]

        np.save(file_path, np.array(new_label))
        print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

        record = [[0 for _ in range(label_num)] for i in range(label_num)]

        for a, b in zip(labels, new_label):
            a, b = int(a), int(b)
            record[a][b] += 1
        #
        print('****************************************')
        print('following is flip percentage:')

        for i in range(label_num):
            sum_i = sum(record[i])
            for j in range(label_num):
                if i != j:
                    print(f"{record[i][j] / sum_i: .2f}", end='\t')
                else:
                    print(f"{record[i][j] / sum_i: .2f}", end='\t')
            print()

        pidx = np.random.choice(range(P.shape[0]), 1000)
        cnt = 0
        for i in range(1000):
            if labels[pidx[i]] == 0:
                a = P[pidx[i], :]
                for j in range(label_num):
                    print(f"{a[j]:.2f}", end="\t")
                print()
                cnt += 1
            if cnt >= 10:
                break
    return torch.LongTensor(new_label)


def noisify_multiclass_symmetric(train_labels, noise_rate, random_state=None, nb_classes=10):
    P = np.ones((nb_classes, nb_classes))
    P = (noise_rate / (nb_classes - 1)) * P

    if noise_rate > 0.0:
        for i in range(nb_classes):
            P[i, i] = 1. - noise_rate

        y_train_noisy = multiclass_noisify(train_labels, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != train_labels).mean()
        assert actual_noise > 0.0
        train_labels = y_train_noisy

    return train_labels


def noisify_pairflip(train_labels, noise_rate, random_state, nb_classes):
    P = np.eye(nb_classes)
    n = noise_rate

    if n > 0.0:
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(train_labels, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != train_labels).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def symmetric_dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    train_labels = train_labels[:, np.newaxis]

    noisy_label = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_seed,
                                               nb_classes=num_classes)

    noisy_label = noisy_label.squeeze()
    clean_labels = train_labels.squeeze()
    num_samples = int(noisy_label.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_label[train_set_index], noisy_label[val_set_index]
    clean_train_labels, clean_val_labels = clean_labels[train_set_index], clean_labels[val_set_index]

    return train_set, train_labels, val_set, val_labels, clean_train_labels, clean_val_labels


def pairflip_dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    train_labels = train_labels[:, np.newaxis]

    noisy_label, real_noise_rate, transition_matrix = noisify_pairflip(train_labels, noise_rate,
                                                                       random_state=random_seed,
                                                                       nb_classes=num_classes)

    noisy_label = noisy_label.squeeze()
    clean_labels = train_labels.squeeze()
    num_samples = int(noisy_label.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_label[train_set_index], noisy_label[val_set_index]
    clean_train_labels, clean_val_labels = clean_labels[train_set_index], clean_labels[val_set_index]

    return train_set, train_labels, val_set, val_labels, clean_train_labels, clean_val_labels


def instance_dataset_split(train_data, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, args=None):
    noise_labels = v6_get_noisy_label(noise_rate, train_data, train_labels, args)
    train_images = train_data.data
    clean_labels = np.array(train_labels)
    num_samples = int(noise_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noise_labels[train_set_index], noise_labels[val_set_index]
    clean_train_labels, clean_val_labels = clean_labels[train_set_index], clean_labels[val_set_index]

    return train_set, train_labels, val_set, val_labels, clean_train_labels, clean_val_labels


def distill_dataset(model, data_loader, data, processed_data, threshold, args, dataset_dir, dataset_type):
    print(f'==> Distilling {dataset_type} set..')
    model.eval()

    distilled_examples_index, distilled_processed_examples_index = [], []
    distilled_examples_labels, distilled_processed_examples_labels = [], []

    for imgs, labels, indexes in data_loader:
        imgs = imgs.to(args.device)
        output, _ = model(imgs)
        pred = torch.max(F.softmax(output, dim=1), 1)
        mask = pred[0] > threshold
        mask = mask.cpu()
        distilled_examples_index.extend(indexes[mask])
        distilled_examples_labels.extend(pred[1].cpu()[mask])
        distilled_processed_examples_index.extend(indexes[~mask])
        distilled_processed_examples_labels.extend(labels[~mask])
    print(f'==> Distilling {dataset_type} set done..')

    # Process and save distilled examples
    distilled_examples_index = np.array(distilled_examples_index)
    distilled_examples_labels = np.array(distilled_examples_labels)
    distilled_processed_examples_index = np.array(distilled_processed_examples_index)
    distilled_processed_examples_labels = np.array(distilled_processed_examples_labels)

    if dataset_type == 'training':
        distilled_imgs = data.train_image[distilled_examples_index]
        distilled_processed_imgs = processed_data.train_image[distilled_processed_examples_index]
        distilled_clean_labels = data.clean_train_label[distilled_examples_index]
    else:
        distilled_imgs = data.val_image[distilled_examples_index]
        distilled_processed_imgs = processed_data.val_image[distilled_processed_examples_index]
        distilled_clean_labels = data.clean_val_label[distilled_examples_index]

    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    distilled_labels = np.concatenate((distilled_examples_labels, distilled_processed_examples_labels), axis=0)

    print(f'Number of distilled {dataset_type} examples: {len(distilled_examples_index)}')
    print(
        f'Accuracy of distilled {dataset_type} examples collection: {(np.array(distilled_examples_labels) == np.array(distilled_clean_labels)).sum() * 100 / len(distilled_examples_labels)}%')

    if dataset_type == 'training':
        np.save(os.path.join(dataset_dir, f'distilled_train_images.npy'), distilled_imgs)
        np.save(os.path.join(dataset_dir, f'distilled_train_labels.npy'), distilled_labels)
    else:
        np.save(os.path.join(dataset_dir, f'distilled_val_images.npy'), distilled_imgs)
        np.save(os.path.join(dataset_dir, f'distilled_val_labels.npy'), distilled_labels)


def distill_dataset_small_loss(train_clean_indices, val_clean_indices, train_data, val_data, train_processed_data,
                               val_processed_data, dataset_dir):
    print(f'==> Filtering dataset..')

    train_distilled_examples_index, train_distilled_processed_examples_index = [], []
    val_distilled_examples_index, val_distilled_processed_examples_index = [], []

    train_distilled_examples_labels, train_distilled_processed_examples_labels = [], []
    val_distilled_examples_labels, val_distilled_processed_examples_labels = [], []

    for i in range(len(train_data.train_label)):
        if i not in train_clean_indices:
            pass
        else:
            train_distilled_examples_index.append(i)
            train_distilled_examples_labels.append(train_data.train_label[i])
        train_distilled_processed_examples_index.append(i)
        train_distilled_processed_examples_labels.append(train_data.train_label[i])

    for i in range(len(val_data.val_label)):
        if i not in val_clean_indices:
            pass
        else:
            val_distilled_examples_index.append(i)
            val_distilled_examples_labels.append(val_data.val_label[i])
        val_distilled_processed_examples_index.append(i)
        val_distilled_processed_examples_labels.append(val_data.val_label[i])

    # make them all numpy arrays
    train_distilled_examples_index = np.array(train_distilled_examples_index)
    train_distilled_examples_labels = np.array(train_distilled_examples_labels)
    train_distilled_processed_examples_index = np.array(train_distilled_processed_examples_index)
    train_distilled_processed_examples_labels = np.array(train_distilled_processed_examples_labels)
    val_distilled_examples_index = np.array(val_distilled_examples_index)
    val_distilled_examples_labels = np.array(val_distilled_examples_labels)
    val_distilled_processed_examples_index = np.array(val_distilled_processed_examples_index)
    val_distilled_processed_examples_labels = np.array(val_distilled_processed_examples_labels)

    # train
    distilled_imgs = train_data.train_image[train_distilled_examples_index]
    distilled_processed_imgs = train_processed_data.train_image[train_distilled_processed_examples_index]
    np.save(os.path.join(dataset_dir, 'train', f'distilled_clean_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'train', f'distilled_clean_labels.npy'), train_distilled_examples_labels)

    distilled_clean_labels = train_data.clean_train_label[train_distilled_examples_index]
    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    # create a classes array which is 0 for distilled examples and 1 for processed distilled examples
    classes = np.concatenate(
        (np.zeros(len(train_distilled_examples_index)), np.ones(len(train_distilled_processed_examples_index))), axis=0)
    distilled_labels = np.concatenate((train_distilled_examples_labels, train_distilled_processed_examples_labels),
                                      axis=0)

    print(f'Number of distilled train examples: {len(train_distilled_examples_index)}')
    print(
        f'Accuracy of distilled train examples collection: {(train_distilled_examples_labels == distilled_clean_labels).sum() * 100 / len(train_distilled_examples_labels)}%')
    np.save(os.path.join(dataset_dir, 'train', f'distilled_train_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'train', f'distilled_train_labels.npy'), distilled_labels)
    np.save(os.path.join(dataset_dir, 'train', f'classes.npy'), classes)

    # val
    distilled_imgs = val_data.val_image[val_distilled_examples_index]
    distilled_processed_imgs = val_processed_data.val_image[val_distilled_processed_examples_index]
    np.save(os.path.join(dataset_dir, 'val', f'distilled_clean_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'val', f'distilled_clean_labels.npy'), val_distilled_examples_labels)
    distilled_clean_labels = val_data.clean_val_label[val_distilled_examples_index]
    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    distilled_labels = np.concatenate((val_distilled_examples_labels, val_distilled_processed_examples_labels), axis=0)

    print(f'Number of distilled validation examples: {len(val_distilled_examples_index)}')
    print(
        f'Accuracy of distilled validation examples collection: {(val_distilled_examples_labels == distilled_clean_labels).sum() * 100 / len(val_distilled_examples_labels)}%')
    np.save(os.path.join(dataset_dir, 'val', f'distilled_val_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'val', f'distilled_val_labels.npy'), distilled_labels)


def distill_dataset_clip(train_clean_indices, val_clean_indices, train_data, val_data, train_processed_data,
                         val_processed_data, dataset_dir):
    print('==> Filtering dataset..')

    train_distilled_examples_index, train_distilled_processed_examples_index = [], []
    val_distilled_examples_index, val_distilled_processed_examples_index = [], []

    train_distilled_examples_labels, train_distilled_processed_examples_labels = [], []
    val_distilled_examples_labels, val_distilled_processed_examples_labels = [], []

    for i in range(len(train_data.train_label)):
        if i not in train_clean_indices:
            pass
        else:
            train_distilled_examples_index.append(i)
            train_distilled_examples_labels.append(train_data.train_label[i])
        train_distilled_processed_examples_index.append(i)
        train_distilled_processed_examples_labels.append(train_data.train_label[i])

    for i in range(len(val_data.val_label)):
        if i not in val_clean_indices:
            pass
        else:
            val_distilled_examples_index.append(i)
            val_distilled_examples_labels.append(val_data.val_label[i])
        val_distilled_processed_examples_index.append(i)
        val_distilled_processed_examples_labels.append(val_data.val_label[i])

    # make them all numpy arrays
    train_distilled_examples_index = np.array(train_distilled_examples_index)
    train_distilled_examples_labels = np.array(train_distilled_examples_labels)
    train_distilled_processed_examples_index = np.array(train_distilled_processed_examples_index)
    train_distilled_processed_examples_labels = np.array(train_distilled_processed_examples_labels)
    val_distilled_examples_index = np.array(val_distilled_examples_index)
    val_distilled_examples_labels = np.array(val_distilled_examples_labels)
    val_distilled_processed_examples_index = np.array(val_distilled_processed_examples_index)
    val_distilled_processed_examples_labels = np.array(val_distilled_processed_examples_labels)

    # train
    distilled_imgs = train_data.train_image[train_distilled_examples_index]
    distilled_processed_imgs = train_processed_data.train_image[train_distilled_processed_examples_index]
    np.save(os.path.join(dataset_dir, 'train', f'distilled_clean_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'train', f'distilled_clean_labels.npy'), train_distilled_examples_labels)

    distilled_clean_labels = train_data.clean_train_label[train_distilled_examples_index]
    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    # create a classes array which is 0 for distilled examples and 1 for processed distilled examples
    classes = np.concatenate(
        (np.zeros(len(train_distilled_examples_index)), np.ones(len(train_distilled_processed_examples_index))), axis=0)
    distilled_labels = np.concatenate((train_distilled_examples_labels, train_distilled_processed_examples_labels),
                                      axis=0)

    print(f'Number of distilled train examples: {len(train_distilled_examples_index)}')
    print(
        f'Accuracy of distilled train examples collection: {(train_distilled_examples_labels == distilled_clean_labels).sum() * 100 / len(train_distilled_examples_labels)}%')
    np.save(os.path.join(dataset_dir, 'train', f'distilled_train_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'train', f'distilled_train_labels.npy'), distilled_labels)
    np.save(os.path.join(dataset_dir, 'train', f'classes.npy'), classes)

    # val
    distilled_imgs = val_data.val_image[val_distilled_examples_index]
    distilled_processed_imgs = val_processed_data.val_image[val_distilled_processed_examples_index]
    np.save(os.path.join(dataset_dir, 'val', f'distilled_clean_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'val', f'distilled_clean_labels.npy'), val_distilled_examples_labels)
    distilled_clean_labels = val_data.clean_val_label[val_distilled_examples_index]
    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    distilled_labels = np.concatenate((val_distilled_examples_labels, val_distilled_processed_examples_labels), axis=0)

    print(f'Number of distilled validation examples: {len(val_distilled_examples_index)}')
    print(
        f'Accuracy of distilled validation examples collection: {(val_distilled_examples_labels == distilled_clean_labels).sum() * 100 / len(val_distilled_examples_labels)}%')
    np.save(os.path.join(dataset_dir, 'val', f'distilled_val_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, 'val', f'distilled_val_labels.npy'), distilled_labels)


def set_up(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if torch.cuda.is_available():
        args.device = 'cuda'
        args.num_workers = 8
        torch.cuda.manual_seed_all(args.seed)
    elif torch.backends.mps.is_available():
        args.device = 'mps'

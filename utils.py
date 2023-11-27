import os

import numpy as np
import torch
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


def v6_get_noisy_label(n, dataset, labels):
    label_num = 10
    norm_std = 0.1
    os.makedirs('./data/cifar10/instance_noise_0.5', exist_ok=True)
    file_path = './data/cifar10/instance_noise_0.5/v6_cifar10_labels_0.5_1.npy'
    print(file_path)
    if os.path.exists(file_path):
        new_label = np.load(file_path)
    else:
        from math import inf
        P = []
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


def instance_dataset_split(train_data, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    noise_labels = v6_get_noisy_label(noise_rate, train_data, train_labels)
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


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


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


def distill_dataset_small_loss(model, train_clean_indices, val_clean_indices, train_data, val_data, train_processed_data,
                               val_processed_data,
                               args, dataset_dir):
    print(f'==> Filtering dataset..')

    train_distilled_examples_index, train_distilled_processed_examples_index = [], []
    val_distilled_examples_index, val_distilled_processed_examples_index = [], []

    train_distilled_examples_labels, train_distilled_processed_examples_labels = [], []
    val_distilled_examples_labels, val_distilled_processed_examples_labels = [], []

    for i in range(len(train_data.train_label)):
        if i not in train_clean_indices:
            train_distilled_processed_examples_index.append(i)
            train_distilled_processed_examples_labels.append(train_data.train_label[i])
        else:
            train_distilled_examples_index.append(i)
            train_distilled_examples_labels.append(train_data.train_label[i])

    for i in range(len(val_data.val_label)):
        if i not in val_clean_indices:
            val_distilled_processed_examples_index.append(i)
            val_distilled_processed_examples_labels.append(val_data.val_label[i])
        else:
            val_distilled_examples_index.append(i)
            val_distilled_examples_labels.append(val_data.val_label[i])

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
    distilled_clean_labels = train_data.clean_train_label[train_distilled_examples_index]
    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    distilled_labels = np.concatenate((train_distilled_examples_labels, train_distilled_processed_examples_labels), axis=0)

    print(f'Number of distilled train examples: {len(train_distilled_examples_index)}')
    print(
        f'Accuracy of distilled train examples collection: {(train_distilled_examples_labels == distilled_clean_labels).sum() * 100 / len(train_distilled_examples_labels)}%')
    np.save(os.path.join(dataset_dir, f'distilled_train_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, f'distilled_train_labels.npy'), distilled_labels)

    # val
    distilled_imgs = val_data.val_image[val_distilled_examples_index]
    distilled_processed_imgs = val_processed_data.val_image[val_distilled_processed_examples_index]
    distilled_clean_labels = val_data.clean_val_label[val_distilled_examples_index]
    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    distilled_labels = np.concatenate((val_distilled_examples_labels, val_distilled_processed_examples_labels), axis=0)

    print(f'Number of distilled validation examples: {len(val_distilled_examples_index)}')
    print(
        f'Accuracy of distilled validation examples collection: {(val_distilled_examples_labels == distilled_clean_labels).sum() * 100 / len(val_distilled_examples_labels)}%')
    np.save(os.path.join(dataset_dir, f'distilled_val_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, f'distilled_val_labels.npy'), distilled_labels)


def source_target_dataset(model, data_loader, data, processed_data, threshold, args, dataset_dir):
    print('==> building source and target dataset..')
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
    print('==> building source and target dataset done..')

    # Process and save distilled examples
    distilled_examples_index = np.array(distilled_examples_index)
    distilled_examples_labels = np.array(distilled_examples_labels)
    distilled_processed_examples_index = np.array(distilled_processed_examples_index)
    distilled_processed_examples_labels = np.array(distilled_processed_examples_labels)

    target_examples_num = len(distilled_processed_examples_index) - len(distilled_examples_index)
    traget_examples_index = np.random.choice(distilled_processed_examples_index, target_examples_num, replace=False)

    distilled_imgs = data.train_image[distilled_examples_index]
    distilled_processed_imgs = processed_data.train_image[distilled_processed_examples_index]
    target_imgs = data.train_image[traget_examples_index]
    distilled_clean_labels = data.clean_train_label[distilled_examples_index]

    distilled_imgs = np.concatenate((distilled_imgs, distilled_processed_imgs), axis=0)
    distilled_labels = np.concatenate((distilled_examples_labels, distilled_processed_examples_labels), axis=0)
    classes = np.concatenate(
        (np.zeros(len(distilled_examples_index)), np.ones(len(distilled_processed_examples_index))), axis=0)

    print(f'Number of distilled training examples: {len(distilled_examples_index)}')
    print(
        f'Accuracy of distilled training examples collection: {(np.array(distilled_examples_labels) == np.array(distilled_clean_labels)).sum() * 100 / len(distilled_examples_labels)}%')
    print(f'Number of target domain examples: {len(traget_examples_index) + len(distilled_examples_index)}')

    np.save(os.path.join(dataset_dir, f'source_images.npy'), distilled_imgs)
    np.save(os.path.join(dataset_dir, f'source_labels.npy'), distilled_labels)
    np.save(os.path.join(dataset_dir, f'target_images.npy'), target_imgs)
    np.save(os.path.join(dataset_dir, f'classes.npy'), classes)


def train(model, train_loader, optimizer, loss_func, args, scheduler):
    model.train()
    train_loss = 0.
    train_acc = 0.
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        output, _ = model(imgs)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.max(F.softmax(output, dim=1), 1)[1]
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()
    scheduler.step()
    return train_loss, train_acc


def train_with_dann(model, source_loader, target_iter, optimizer, loss_func, domain_adv, args, scheduler):
    model.train()
    domain_adv.train()
    train_loss = 0.
    train_acc = 0.
    domain_acc = 0.
    for imgs, labels, classes, _ in source_loader:
        source_num = int(classes.sum().item())
        sorted_indices = torch.argsort(classes, descending=True)
        imgs, labels = imgs[sorted_indices], labels[sorted_indices]
        if source_num - (len(imgs) - source_num) > 0:
            target_imgs = target_iter.get_data(source_num - (len(imgs) - source_num))
            imgs, labels, target_imgs = imgs.to(args.device), labels.to(args.device), target_imgs.to(args.device)
            input = torch.cat((imgs, target_imgs), dim=0)
        else:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            input = imgs
        output, features = model(input)
        source_output = output[:len(imgs)]
        source_features, target_features = features[:source_num], features[-source_num:]
        cls_loss = loss_func(source_output, labels)
        transfer_loss = domain_adv(source_features, target_features)
        loss = cls_loss + transfer_loss * args.trade_off
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.max(F.softmax(source_output, dim=1), 1)[1]
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()  # account
        domain_acc += domain_adv.domain_discriminator_accuracy.item()  # %
    scheduler.step()
    return train_loss, train_acc, domain_acc


def train_with_mdd(model, adv, source_loader, target_iter, optimizer, loss_func, mdd, args, scheduler):
    model.train()
    mdd.train()
    adv.train()
    train_loss = 0.
    train_acc = 0.
    for imgs, labels, classes, _ in source_loader:
        source_num = int(classes.sum().item())
        img_num = len(imgs)
        sorted_indices = torch.argsort(classes, descending=True)
        imgs, labels = imgs[sorted_indices], labels[sorted_indices]
        if source_num - (img_num - source_num) > 0:
            target_imgs = target_iter.get_data(source_num - (img_num - source_num))
            imgs, labels, target_imgs = imgs.to(args.device), labels.to(args.device), target_imgs.to(args.device)
            input = torch.cat((imgs, target_imgs), dim=0)
        else:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            input = imgs
        output, features = model(input)
        outputs_adv = adv(features)
        source_output = output[:img_num]
        source_adv, target_adv = outputs_adv[:source_num], outputs_adv[-source_num:]
        y_s, y_t = output[:source_num], output[-source_num:]
        cls_loss = loss_func(source_output, labels)
        transfer_loss = -mdd(y_s, source_adv, y_t, target_adv)
        loss = cls_loss + transfer_loss * args.trade_off
        optimizer.zero_grad()
        adv.step()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.max(F.softmax(source_output, dim=1), 1)[1]
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()
    scheduler.step()
    return train_loss, train_acc
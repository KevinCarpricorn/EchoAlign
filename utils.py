import os

import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal


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


def noisify_multiclass_symmetric(train_labels, noise_rate, random_state=None, nb_classes=10):
    P = np.ones((nb_classes, nb_classes))
    P = (noise_rate / (nb_classes - 1)) * P

    if noise_rate >0.0:
        for i in range(nb_classes):
            P[i, i] = 1. - noise_rate

        y_train_noisy = multiclass_noisify(train_labels, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != train_labels).mean()
        assert actual_noise > 0.0
        train_labels = y_train_noisy

    return  train_labels


def dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    train_labels = train_labels[:, np.newaxis]

    noisy_label = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_seed, nb_classes=num_classes)

    noisy_label = noisy_label.squeeze()
    clean_labels = train_labels.squeeze()
    num_samples = int(noisy_label.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_label[train_set_index], noisy_label[val_set_index]
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
    print(f'Accuracy of distilled {dataset_type} examples collection: {(np.array(distilled_examples_labels) == np.array(distilled_clean_labels)).sum() * 100 / len(distilled_examples_labels)}%')

    if dataset_type == 'training':
        np.save(os.path.join(dataset_dir, f'distilled_train_images.npy'), distilled_imgs)
        np.save(os.path.join(dataset_dir, f'distilled_train_labels.npy'), distilled_labels)
    else:
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
    classes = np.concatenate((np.zeros(len(distilled_examples_index)), np.ones(len(distilled_processed_examples_index))), axis=0)

    print(f'Number of distilled training examples: {len(distilled_examples_index)}')
    print(f'Accuracy of distilled training examples collection: {(np.array(distilled_examples_labels) == np.array(distilled_clean_labels)).sum() * 100 / len(distilled_examples_labels)}%')
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
        train_acc += train_correct.item() # account
        domain_acc += domain_adv.domain_discriminator_accuracy.item() # %
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
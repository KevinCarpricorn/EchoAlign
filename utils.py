import numpy as np
import torch
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
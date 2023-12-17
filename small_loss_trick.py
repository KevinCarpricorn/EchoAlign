import numpy as np
import torch
from tqdm import tqdm


def train(model, train_loader, val_loader, optimizer, loss_fn, args):
    print("==> Warmming up the model...")
    train_loss_matrix = np.zeros((len(train_loader.dataset), args.distill_epochs))
    val_loss_matrix = np.zeros((len(val_loader.dataset), args.distill_epochs))
    for epoch in tqdm(range(args.distill_epochs)):
        model.train()
        print("==> Epoch: {}".format(epoch + 1))
        for imgs, labels, indices in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            outputs, _ = model(imgs)
            loss = loss_fn(outputs, labels)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch != args.distill_epochs:
            train_loss_matrix, val_loss_matrix = eval(model, train_loader, val_loader,
                                                      torch.nn.CrossEntropyLoss(reduction='none'), train_loss_matrix,
                                                      val_loss_matrix,
                                                      epoch, args)
    return train_loss_matrix, val_loss_matrix


def eval(model, train_loader, val_loader, loss_fn, train_loss_matrix, val_loss_matrix, epoch, args):
    model.eval()
    with torch.no_grad():
        for imgs, labels, indices in val_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            outputs, _ = model(imgs)
            loss = loss_fn(outputs, labels)
            val_loss_matrix[indices, epoch] = loss.detach().cpu().numpy()
        for imgs, labels, indices in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            outputs, _ = model(imgs)
            loss = loss_fn(outputs, labels)
            train_loss_matrix[indices, epoch] = loss.detach().cpu().numpy()
    return train_loss_matrix, val_loss_matrix


def filter(model, train_loader, val_loader, train_data, val_data, optimizer, loss_fn, args):
    train_loss_matrix, val_loss_matrix = train(model, train_loader, val_loader, optimizer, loss_fn, args)
    train_loss_matrix = train_loss_matrix.mean(axis=1)
    val_loss_matrix = val_loss_matrix.mean(axis=1)

    cr = min(1 - 1.2 * args.noise_rate, 0.9 * (1 - args.noise_rate))
    train_sorted_indices = np.argsort(train_loss_matrix)
    val_sorted_indices = np.argsort(val_loss_matrix)

    train_clean_indices = []
    val_clean_indices = []
    for cls in range(args.num_classes):
        c = []
        for i in train_sorted_indices:
            if train_data.train_label[i] == cls:
                c.append(i)
        clean_num = int(cr * len(c))
        train_clean_indices.extend(c[:clean_num])

        c = []
        for i in val_sorted_indices:
            if val_data.val_label[i] == cls:
                c.append(i)
        clean_num = int(cr * len(c))
        val_clean_indices.extend(c[:clean_num])

    return train_clean_indices, val_clean_indices

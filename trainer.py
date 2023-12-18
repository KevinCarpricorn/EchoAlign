import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluator import test


def train(model, train_loader, optimizer, loss_func, args, scheduler, source_weight, target_weight):
    model.train()
    train_loss = 0.
    train_acc = 0.
    for imgs, labels, cls, _ in train_loader:
        sorted_indices = torch.argsort(cls, descending=True)
        imgs, labels = imgs[sorted_indices], labels[sorted_indices]
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        weights = torch.tensor([source_weight] * int(cls.sum()) + [target_weight] * (len(cls) - int(cls.sum())))
        weights = weights.to(args.device)
        output, _ = model(imgs)
        loss = loss_func(output, labels)
        loss = torch.mean(loss * weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.max(F.softmax(output, dim=1), 1)[1]
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()
    scheduler.step()
    return train_loss, train_acc


def fine_tune(model, fine_tune_loader, test_loader, optimizer, loss_func, args):
    best_acc = 0.
    for epoch in tqdm(range(40)):
        model.train()
        train_loss = 0.
        train_acc = 0.
        for imgs, labels, _ in fine_tune_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output, _ = model(imgs)
            loss = loss_func(output, labels)
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.max(F.softmax(output, dim=1), 1)[1]
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()
        print('Fine Tune Loss: {:.6f}, Acc: {:.6f}%'.format(
            train_loss / (len(fine_tune_loader.dataset)) * args.batch_size,
            train_acc * 100 / (len(fine_tune_loader.dataset))))

        test_loss, test_acc = test(model, test_loader, loss_func, args)
        if test_acc > best_acc:
            best_acc = test_acc
        print('Test Loss: {:.6f}, Acc: {:.6f}%'.format(test_loss / (len(test_loader.dataset)) * args.batch_size,
                                                       test_acc * 100 / (len(test_loader.dataset))))
    return best_acc
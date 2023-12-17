import torch
import torch.nn.functional as F


def evaluate(model, val_loader, loss_func, args):
    model.eval()
    val_loss = 0.
    val_acc = 0.
    with torch.no_grad():
        for imgs, labels, _, _ in val_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output, _ = model(imgs)
            loss = loss_func(output, labels)
            loss = loss.mean()
            val_loss += loss.item()
            pred = torch.max(F.softmax(output, dim=1), 1)[1]
            val_correct = (pred == labels).sum()
            val_acc += val_correct.item()
    return val_loss, val_acc


def test(model, test_loader, loss_func, args):
    model.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output, _ = model(imgs)
            loss = loss_func(output, labels)
            loss = loss.mean()
            test_loss += loss.item()
            pred = torch.max(F.softmax(output, dim=1), 1)[1]
            test_correct = (pred == labels).sum()
            test_acc += test_correct.item()
    return test_loss, test_acc

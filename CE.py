import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
import logs.logger as logger
import models
import transforms
from args_parser import parse_args
from evaluator import test
from utils import *

args = parse_args()
# set up environment
set_up(args)

print(
    f'==> Dataset: {args.dataset}, Noise Type: {args.noise_type}, Noise Rate: {args.noise_rate}, Batch Size: {args.batch_size}, Seed: {args.seed}')

# preparing dataset
transform = transforms.transform(args)
target_transform = transforms.target_transform

print('==> Preparing data..')
# data loading
train_data = dataloader.get_noisy_dataset(train=True, transform=transform, target_transform=target_transform, args=args,
                                          exist=args.exist)
val_data = dataloader.get_noisy_dataset(train=False, transform=transform, target_transform=target_transform, args=args,
                                        exist=args.exist)

# data loader
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=False)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=False)
test_loader = dataloader.get_test_loader(transform, target_transform, args, exist=args.exist)

# set up model
model = models.get_model(args).to(args.device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# loss
loss_func = nn.CrossEntropyLoss().to(args.device)

print('==> Start training..')


def main():
    best_acc = 0.
    for epoch in tqdm(range(args.n_epoch)):
        print('epoch {}'.format(epoch + 1))
        # train
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

        # evaluate
        model.eval()
        val_loss = 0.
        val_acc = 0.
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                output, _ = model(imgs)
                loss = loss_func(output, labels)
                val_loss += loss.item()
                pred = torch.max(F.softmax(output, dim=1), 1)[1]
                val_correct = (pred == labels).sum()
                val_acc += val_correct.item()

        print('Train Loss: {:.6f}, Acc: {:.6f}%'.format(train_loss / (len(train_data)) * args.batch_size,
                                                        train_acc * 100 / (len(train_data))))
        print('Val Loss: {:.6f}, Acc: {:.6f}%'.format(val_loss / (len(val_data)) * args.batch_size,
                                                      val_acc * 100 / (len(val_data))))

        test_loss, test_acc = test(model, test_loader, loss_func, args)
        if test_acc > best_acc:
            best_acc = test_acc
        print('Test Loss: {:.6f}, Acc: {:.6f}%'.format(test_loss / (len(test_loader.dataset)) * args.batch_size,
                                                       test_acc * 100 / (len(test_loader.dataset))))
    print('Best Test Acc: {:.6f}%'.format(best_acc * 100 / (len(test_loader.dataset))))


if __name__ == '__main__':
    log_dir = f'./logs/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    logger.log(main, os.path.join(log_dir, f'{args.noise_type}_{args.noise_rate}.log'))

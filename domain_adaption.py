import itertools

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
import dataset
import logs.logger as logger
import models
import transforms
from args_parser import parse_args
from evaluator import test
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from utils import *

args = parse_args()
# set up environment
set_up(args)

# preparing dataset
transform = transforms.transform(args)
target_transform = transforms.target_transform

print('==> Preparing data..')
# data loading
target_data = dataloader.get_noisy_dataset(train=True, transform=transform, target_transform=target_transform,
                                           args=args,
                                           exist=True)
source_data = dataloader.get_processed_dataset(train=True, transform=transform,
                                               target_transform=target_transform, args=args, exist=True)

# data loader
source_loader = DataLoader(source_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                           drop_last=False)
target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                           drop_last=False)
target_iter = dataset.CustomDataIterator(target_loader)
test_loader = dataloader.get_test_loader(transform, target_transform, args, exist=True)

# set up model
model = models.get_model(args).to(args.device)
domain_discri = DomainDiscriminator(in_feature=args.features_dim, hidden_size=1024).to(args.device)
optimizer = optim.SGD(itertools.chain(model.parameters(), domain_discri.parameters()), lr=args.lr,
                      weight_decay=args.weight_decay, momentum=0.9)
domain_adv = DomainAdversarialLoss(domain_discri).to(args.device)

# loss
loss_func = nn.CrossEntropyLoss().to(args.device)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

print('==> Start training..')


def main():
    best_acc = 0.
    for epoch in tqdm(range(args.n_epoch)):
        print('epoch {}'.format(epoch + 1))
        # train
        model.train()
        domain_adv.train()
        train_loss = 0.
        train_acc = 0.
        domain_acc = 0.
        for imgs, labels, _ in source_loader:
            target_imgs, _, _ = target_iter.get_data(1)
            imgs, labels, target_imgs = imgs.to(args.device), labels.to(args.device), target_imgs.to(args.device)
            input = torch.cat((imgs, target_imgs), dim=0)
            output, features = model(input)
            source_output = output[:len(imgs)]
            source_features, target_features = features.chunk(2, dim=0)
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
            domain_acc += domain_adv.domain_discriminator_accuracy.item()
        scheduler.step()

        print('Train Loss: {:.6f}, Acc: {:.6f}%, Domain Acc: {:.6f}%'.format(
            train_loss / (len(source_data)) * args.batch_size,
            train_acc * 100 / (len(source_data)),
            domain_acc / (len(source_loader))))

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

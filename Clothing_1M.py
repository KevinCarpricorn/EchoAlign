import os.path
import sys
import tarfile

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
from clip_filter import filter_clothing1m
from evaluator import evaluate, test
from trainer import train, fine_tune
from utils import *

args = parse_args()
# set up environment
set_up(args)

print(
    f'==> Dataset: {args.dataset}, Batch Size: {args.batch_size}, Seed: {args.seed}, Learning Rate: {args.lr}, Weight Decay: {args.weight_decay}, threshold: {args.threshold}')

# preparing dataset
transform = transforms.transform(args)
target_transform = transforms.target_transform

print('==> Preparing data..')
# data loading
job_path = os.getenv('PBS_JOBFS')
data_path = ['noisy.tar', 'clean_val.tar', 'clean_test.tar']
data_dir = '/scratch/yc49/yz4497/neurips/data/Clothing1M/base/images'
for tar_file in data_path:
    tar_file = os.path.join(data_dir, tar_file)
    with tarfile.open(tar_file, 'r') as tar:
        tar.extractall(path=job_path)
tar_file = '/scratch/yc49/yz4497/neurips/data/Clothing1M/base/processed_images/processed_noisy.tar'
with tarfile.open(tar_file, 'r') as tar:
    tar.extractall(path=job_path)

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

# loss
loss_func = nn.CrossEntropyLoss(reduction='none').to(args.device)

# model saving directory
model_save_dir = os.path.join(args.model_dir, args.dataset)
model_file = f'{args.model}_best_model.pth'
os.makedirs(model_save_dir, exist_ok=True)

# distillation
print('==> Distilled dataset building..')
distilled_dataset_dir = os.path.join('./data', args.dataset, 'base')

if os.path.exists(os.path.join(distilled_dataset_dir, 'distilled_clean_images.tar')):
    print('==> Distilled dataset exists..')
else:
    train_clean_indices = filter_clothing1m(args)
    distill_dataset_Clothing1M(train_clean_indices, train_data)

train_data = dataloader.get_distilled_dataset(train=True, transform=transform, target_transform=target_transform,
                                              dir=distilled_dataset_dir, args=args)
val_data = dataloader.get_distilled_dataset(train=False, transform=transform, target_transform=target_transform,
                                            dir=distilled_dataset_dir, args=args)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=False, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=False, pin_memory=True)
# renew model
model = models.get_model(args).to(args.device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
print('==> Distilled dataset building done..')

print('==> Start training..')


def main():
    best_val_acc = 0.
    source_weight = 1000000 / len(train_data)
    target_weight = (len(train_data) - 1000000) / len(train_data)
    print('source_weight: {:.6f}, target_weight: {:.6f}'.format(source_weight, target_weight))
    best_acc = 0.
    for epoch in tqdm(range(args.n_epoch)):
        print('epoch {}'.format(epoch + 1))
        # train
        train_loss, train_acc = train(model, train_loader, optimizer, loss_func, args, scheduler, source_weight,
                                      target_weight)

        # evaluate
        val_loss, val_acc = evaluate(model, val_loader, loss_func, args)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, model_file))

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

    # model.load_state_dict(torch.load(model_save_dir + '/' + model_file))
    # test_loss, test_acc = test(model, test_loader, loss_func, args)
    # print('Test Loss: {:.6f}, Acc: {:.6f}%'.format(test_loss / (len(test_loader.dataset)) * args.batch_size,
    #                                                test_acc * 100 / (len(test_loader.dataset))))

    # fine tune
    # best_acc = fine_tune(model, fine_tune_loader, test_loader, optimizer, loss_func, args)
    print('Best Test Acc: {:.6f}%'.format(best_acc * 100 / (len(test_loader.dataset))))


if __name__ == '__main__':
    log_dir = f'./logs/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    logger.log(main, os.path.join(log_dir, f'{args.dataset}_{args.seed}.log'))

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
from evaluator import evaluate, test
from small_loss_trick import filter
from clip_filter import filtering
from trainer import train, fine_tune
from utils import *

args = parse_args()
# set up environment
set_up(args)

print(f'==> Dataset: {args.dataset}, Noise Type: {args.noise_type}, Noise Rate: {args.noise_rate}, Batch Size: {args.batch_size}, Seed: {args.seed}')

# preparing dataset
transform = transforms.transform(args)
target_transform = transforms.target_transform

print('==> Preparing data..')
# data loading
train_data = dataloader.get_noisy_dataset(train=True, transform=transform, target_transform=target_transform, args=args,
                                          exist=args.exist)
val_data = dataloader.get_noisy_dataset(train=False, transform=transform, target_transform=target_transform, args=args,
                                        exist=args.exist)
processed_train_data = dataloader.get_processed_dataset(train=True, transform=transform,
                                                        target_transform=target_transform, args=args, exist=args.exist)
processed_val_data = dataloader.get_processed_dataset(train=False, transform=transform,
                                                      target_transform=target_transform, args=args, exist=args.exist)

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
model_save_dir = os.path.join(args.model_dir, args.dataset, f'{args.noise_type}_{args.noise_rate}')
model_file = f'{args.model}_best_model.pth'
os.makedirs(model_save_dir, exist_ok=True)

# distillation
print('==> Distilled dataset building..')
distilled_dataset_dir = os.path.join('./data', args.dataset, f'{args.noise_type}_{args.noise_rate}')

if args.noise_type == 'symmetric':
    train_clean_indices, val_clean_indices = filter(model, train_loader, val_loader, train_data, val_data, optimizer,
                                                    loss_func, args)
    distill_dataset_small_loss(train_clean_indices, val_clean_indices, train_data, val_data, processed_train_data,
                               processed_val_data, distilled_dataset_dir)
elif args.noise_type == 'instance':
    train_clean_indices, val_clean_indices = filtering(args)
    distill_dataset_clip(train_clean_indices, val_clean_indices, train_data, val_data, processed_train_data,
                            processed_val_data, distilled_dataset_dir)
elif args.noise_type == 'pairflip':
    # clip
    train_clean_indices, val_clean_indices = filtering(args)
    distill_dataset_clip(train_clean_indices, val_clean_indices, train_data, val_data, processed_train_data,
                         processed_val_data, distilled_dataset_dir)

train_data = dataloader.get_distilled_dataset(train=True, transform=transform, target_transform=target_transform,
                                              dir=distilled_dataset_dir, args=args)
val_data = dataloader.get_distilled_dataset(train=False, transform=transform, target_transform=target_transform,
                                            dir=distilled_dataset_dir, args=args)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=False)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=False)
# renew model
model = models.get_model(args).to(args.device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
fine_tune_data = dataset.filtered_dataset(train=True, transform=transform, target_transform=target_transform,
                                          dir=distilled_dataset_dir)
fine_tune_loader = DataLoader(fine_tune_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=False)
print('==> Distilled dataset building done..')

print('==> Start training..')


def main():
    best_val_acc = 0.
    source_weight = len(train_clean_indices) / len(train_data)
    target_weight = (len(train_data) - len(train_clean_indices)) / len(train_data)
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
    best_acc = fine_tune(model, fine_tune_loader, test_loader, optimizer, loss_func, args)
    print('Best Test Acc: {:.6f}%'.format(best_acc * 100 / (len(test_loader.dataset))))


if __name__ == '__main__':
    log_dir = f'./logs/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    logger.log(main, os.path.join(log_dir, f'{args.noise_type}_{args.noise_rate}.log'))

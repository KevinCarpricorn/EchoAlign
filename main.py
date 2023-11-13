import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
import torchvision.transforms as transforms
import models
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils import transform_target, distill_dataset
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model')
parser.add_argument('--prob_dir', type=str, help='dir to save output probability files', default='prob')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--processed', type=bool, default=True)
parser.add_argument('--distill', type=bool, default=True)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--warmup_epochs', type=int, default=30)
# set to 0 if using cpu
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

if args.distill:
    args.processed = False

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

if torch.cuda.is_available():
    args.device = 'cuda'
    args.num_workers = 8


# dataset(cifar10)
args.num_classes = 10
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print('==> Preparing data..')
if args.processed:
    train_data = dataset.processed_CIFAR10(train=True, transform=transform, target_transform=transform_target)
    val_data = dataset.processed_CIFAR10(train=False, transform=transform, target_transform=transform_target)
elif args.distill:
    train_data = dataset.CIFAR10(train=True, transform=transform, target_transform=transform_target,
                                 noise_rate=args.noise_rate, random_seed=args.seed)
    val_data = dataset.CIFAR10(train=False, transform=transform, target_transform=transform_target,
                               noise_rate=args.noise_rate, random_seed=args.seed)
    processed_train_data = dataset.processed_CIFAR10(train=True, transform=transform, target_transform=transform_target)
    processed_val_data = dataset.processed_CIFAR10(train=False, transform=transform, target_transform=transform_target)
else:
    train_data = dataset.CIFAR10(train=True, transform=transform, target_transform=transform_target,
                                 noise_rate=args.noise_rate, random_seed=args.seed)
    val_data = dataset.CIFAR10(train=False, transform=transform, target_transform=transform_target,
                               noise_rate=args.noise_rate, random_seed=args.seed)
test_data = dataset.CIFAR10_test(transform=transform, target_transform=transform_target)

# Data Loader
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=False)
distill_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          drop_last=False)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         drop_last=False)

model = models.ResNet18(args.num_classes)
model = model.to(args.device)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# loss
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(args.device)

# model saving directory
model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'noise_rate_%s' % (args.noise_rate)
distill_model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'warm_up'
if not os.path.exists(model_save_dir):
    os.system('mkdir -p %s' % (model_save_dir))
if not os.path.exists(distill_model_save_dir):
    os.system('mkdir -p %s' % (distill_model_save_dir))

# Warmup for distillation
if args.distill:
    print('==> Warmup for distillation..')
    best_acc = 0.
    for epoch in tqdm(range(args.warmup_epochs)):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output = model(imgs)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = 0.
        with torch.no_grad():
            model.eval()
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                output = model(imgs)
                pred = torch.max(F.softmax(output, dim=1), 1)[1]
                val_correct = (pred == labels).sum()
                val_acc += val_correct.item()
            print(

                f'Warm up Epoch {epoch + 1} Validation Accuracy on the {len(val_data)} test data: {val_acc * 100 / (len(val_data)):.4f}')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), distill_model_save_dir + '/' + 'best_model.pth')

    # distillation
    threshold = (1 + args.rho) / 2
    model.load_state_dict(torch.load(distill_model_save_dir + '/' + 'best_model.pth'))
    distilled_dataset_dir = os.path.join('./data', args.dataset, 'distilled_dataset')
    os.makedirs(distilled_dataset_dir, exist_ok=True)

    distill_dataset(model, distill_train_loader, train_data, processed_train_data, threshold, args,
                    distilled_dataset_dir, "training")
    distill_dataset(model, val_loader, val_data, processed_val_data, threshold, args, distilled_dataset_dir,
                    "validation")

print('==> Start training..')
def mian():
    best_val_acc = 0.
    for epoch in tqdm(range(args.n_epoch)):
        print('epoch {}'.format(epoch + 1))
        # train
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output = model(imgs)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.max(F.softmax(output, dim=1), 1)[1]
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                output = model(imgs)
                loss = loss_func(output, labels)
                val_loss += loss.item()
                pred = torch.max(F.softmax(output, dim=1), 1)[1]
                val_correct = (pred == labels).sum()
                val_acc += val_correct.item()

        if args.processed:
            model_file = 'processed_best_model.pth'
        else:
            model_file = 'best_model.pth'

        if val_acc * 100 / (len(val_data)) > best_val_acc:
            best_val_acc = val_acc * 100 / (len(val_data))
            torch.save(model.state_dict(), model_save_dir + '/' + model_file)

        print('Train Loss: {:.6f}, Acc: {:.6f}%'.format(train_loss / (len(train_data)) * args.batch_size,
                                                        train_acc * 100 / (len(train_data))))
        print('Val Loss: {:.6f}, Acc: {:.6f}%'.format(val_loss / (len(val_data)) * args.batch_size,
                                                      val_acc * 100 / (len(val_data))))

    test_loss = 0.
    test_acc = 0.
    model.load_state_dict(torch.load(model_save_dir + '/' + model_file))
    with torch.no_grad():
        model.eval()
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output = model(imgs)
            loss = loss_func(output, labels)
            test_loss += loss.item()
            pred = torch.max(F.softmax(output, dim=1), 1)[1]
            test_correct = (pred == labels).sum()
            test_acc += test_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}%'.format(test_loss / (len(test_data)) * args.batch_size,
                                                   test_acc * 100 / (len(test_data))))


if __name__ == '__main__':
    mian()

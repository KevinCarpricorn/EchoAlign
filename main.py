import itertools

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import models
from args_parser import parse_args
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from utils import *
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy \
    as MarginDisparityDiscrepancy

args = parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

if torch.cuda.is_available():
    args.device = 'cuda'
    args.num_workers = 8
    torch.cuda.manual_seed_all(args.seed)

# dataset(cifar10)
args.num_classes = 10
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print('==> Preparing data..')
# data loading
if args.mode == 'processed_only':
    dataset_function = dataset.processed_CIFAR10
elif args.mode == 'distill_only' or args.mode == 'all':
    dataset_function = dataset.CIFAR10
    processed_train_data = dataset.processed_CIFAR10(train=True, transform=transform, target_transform=transform_target)
    processed_val_data = dataset.processed_CIFAR10(train=False, transform=transform, target_transform=transform_target)
elif args.mode == 'raw_only':
    dataset_function = dataset.CIFAR10

if args.mode == 'processed_only':
    train_data = dataset_function(train=True, transform=transform, target_transform=transform_target)
    val_data = dataset_function(train=False, transform=transform, target_transform=transform_target)
else:
    train_data = dataset_function(train=True, transform=transform, target_transform=transform_target,
                                  noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    val_data = dataset_function(train=False, transform=transform, target_transform=transform_target,
                                noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)

test_data = dataset.CIFAR10_test(transform=transform, target_transform=transform_target)

# data loader
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=False)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        drop_last=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         drop_last=False)

if args.mode == 'distill_only' or args.mode == 'all':
    distill_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, drop_last=False)

# set up model
model = models.ResNet18(args.num_classes)
# domain_discri = models.adversarial(512, 1024, 1024, args.num_classes).to(args.device)
model = model.to(args.device)
if args.mode == 'all':
    domain_discri = DomainDiscriminator(in_feature=args.features_dim, hidden_size=1024).to(args.device)
    optimizer = optim.SGD(itertools.chain(model.parameters(), domain_discri.parameters()), lr=args.lr,
                          weight_decay=args.weight_decay, momentum=0.9)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# loss
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(args.device)
if args.mode == 'all':
    domain_adv = DomainAdversarialLoss(domain_discri).to(args.device)
    # mdd = MarginDisparityDiscrepancy(4.).to(args.device)

# model saving directory
model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'noise_rate_%s' % (args.noise_rate)
distill_model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'warm_up'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(distill_model_save_dir, exist_ok=True)

print('==> Warmup for distillation..')
warm_up_model = os.path.join(distill_model_save_dir, 'best_model.pth')
if not os.path.exists(warm_up_model):
    best_acc = 0.
    for epoch in tqdm(range(args.warmup_epochs)):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            output, _ = model(imgs)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = 0.
        with torch.no_grad():
            model.eval()
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                output, _ = model(imgs)
                pred = torch.max(F.softmax(output, dim=1), 1)[1]
                val_correct = (pred == labels).sum()
                val_acc += val_correct.item()
            print(
                f'Warm up Epoch {epoch + 1} Validation Accuracy on the {len(val_data)} test data: {val_acc * 100 / (len(val_data)):.4f}')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), warm_up_model)

threshold = (1 + args.rho) / 2
if args.mode != 'raw_only' and args.mode != 'processed_only':
    model.load_state_dict(torch.load(warm_up_model))

# distillation
if args.mode == 'distill_only':
    # distillation
    distilled_dataset_dir = os.path.join('./data', args.dataset, 'distilled_dataset')
    os.makedirs(distilled_dataset_dir, exist_ok=True)

    distill_dataset(model, distill_train_loader, train_data, processed_train_data, threshold, args,
                    distilled_dataset_dir, "training")
    distill_dataset(model, val_loader, val_data, processed_val_data, threshold, args, distilled_dataset_dir,
                    "validation")

    print('==> Distilled dataset building..')
    train_data = dataset.distilled_CIFAR10(train=True, transform=transform, target_transform=transform_target)
    val_data = dataset.distilled_CIFAR10(train=False, transform=transform, target_transform=transform_target)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=False)
    print('==> Distilled dataset building done..')
elif args.mode == 'all':
    tf_dataset_dir = os.path.join('./data', args.dataset, 'source_target_dataset')
    os.makedirs(tf_dataset_dir, exist_ok=True)

    # source_target_dataset(model, distill_train_loader, train_data, processed_train_data, threshold, args,
    #                       tf_dataset_dir)
    # distill_dataset(model, val_loader, val_data, processed_val_data, threshold, args, tf_dataset_dir,
    #                 "validation")

    print('==> Source and target dataset building..')
    source_data = dataset.source_CIFAR10(transform=transform, target_transform=transform_target)
    target_data = dataset.target_CIFAR10(transform=transform)
    val_data = dataset.distilled_CIFAR10(train=False, transform=transform, target_transform=transform_target, dir=tf_dataset_dir)
    source_loader = DataLoader(source_data, batch_size=int(args.batch_size / 2), shuffle=True,
                               num_workers=args.num_workers,
                               drop_last=False)
    target_loader = DataLoader(target_data, batch_size=1, shuffle=True,
                               num_workers=args.num_workers,
                               drop_last=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=False)
    target_iter = dataset.CustomDataIterator(target_loader)
    print('==> Source and target dataset building done..')

print('==> Start training..')


def mian():
    best_val_acc = 0.
    for epoch in tqdm(range(args.n_epoch)):
        print('epoch {}'.format(epoch + 1))
        # train
        val_loss = 0.
        val_acc = 0.
        if args.mode == 'all':
            train_loss, train_acc, domain_acc = train_with_dann(model, source_loader, target_iter, optimizer, loss_func,
                                                                domain_adv, args, scheduler)
            # train_loss, train_acc = train_with_mdd(model, domain_discri, source_loader, target_iter, optimizer, loss_func, mdd, args, scheduler)
        else:
            train_loss, train_acc = train(model, train_loader, optimizer, loss_func, args, scheduler)

        with torch.no_grad():
            model.eval()
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                output, _ = model(imgs)
                loss = loss_func(output, labels)
                val_loss += loss.item()
                pred = torch.max(F.softmax(output, dim=1), 1)[1]
                val_correct = (pred == labels).sum()
                val_acc += val_correct.item()

        if args.mode == 'distill_only':
            model_file = 'distilled_best_model.pth'
        elif args.mode == 'processed_only':
            model_file = 'processed_best_model.pth'
        elif args.mode == 'raw_only':
            model_file = 'best_model.pth'
        else:
            model_file = 'dann_best_model.pth'

        if val_acc * 100 / (len(val_data)) > best_val_acc:
            best_val_acc = val_acc * 100 / (len(val_data))
            torch.save(model.state_dict(), model_save_dir + '/' + model_file)

        if args.mode == 'all':
            print('Train Loss: {:.6f}, Acc: {:.6f}%, Domain Acc: {:.6f}%'.format(
                train_loss / (len(train_data)) * args.batch_size,
                train_acc * 100 / (len(train_data)),
                domain_acc / (len(source_loader))))
        else:
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
            output, _ = model(imgs)
            loss = loss_func(output, labels)
            test_loss += loss.item()
            pred = torch.max(F.softmax(output, dim=1), 1)[1]
            test_correct = (pred == labels).sum()
            test_acc += test_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}%'.format(test_loss / (len(test_data)) * args.batch_size,
                                                   test_acc * 100 / (len(test_data))))


if __name__ == '__main__':
    mian()

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
    parser.add_argument('--model_dir', type=str, help='dir to save model files', default='./model')
    parser.add_argument('--prob_dir', type=str, help='dir to save output probability files', default='prob')
    parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='cifar10')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.45)
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distill_epochs', type=int, default=70)
    parser.add_argument('--features_dim', type=int, default=512)
    parser.add_argument('--trade_off', type=float, default=1.0)
    parser.add_argument('--noise_type', type=str, default='pairflip', choices=['symmetric', 'pairflip', 'instance'])
    parser.add_argument('--exist', type=bool, default=True)
    # set to 0 if using cpu
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

import os

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to('cuda')


def list_files(base_path):
    all_files = []
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        files = [f for f in files if not f.startswith('.')]
        dirs.sort()
        files.sort()
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_indices(threshold=0.5, images=None, processed_images=None):
    indices = []
    for i in tqdm(range(len(images))):
        image = Image.fromarray(images[i])
        processed_image = Image.fromarray(processed_images[i])
        image = preprocess(image).unsqueeze(0)
        processed_image = preprocess(processed_image).unsqueeze(0)
        image = image.to('cuda')
        processed_image = processed_image.to('cuda')
        with torch.no_grad():
            image_features = model.encode_image(image)
            processed_image_features = model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            processed_image_features /= processed_image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ processed_image_features.T
        if similarity > threshold:
            indices.append(i)
    return indices


def get_indices_Clothing1M(threshold=0.5):
    indices = []
    job_path = os.getenv('PBS_JOBFS')
    images_path = os.path.join(job_path, 'noisy')
    processed_images_path = os.path.join(job_path, 'processed_noisy')
    images_path = list_files(images_path)
    processed_images_path = list_files(processed_images_path)
    for i in tqdm(range(len(images_path))):
        image = Image.open(images_path[i])
        processed_image = Image.open(processed_images_path[i])
        image = preprocess(image).unsqueeze(0)
        processed_image = preprocess(processed_image).unsqueeze(0)
        image = image.to('cuda')
        processed_image = processed_image.to('cuda')
        with torch.no_grad():
            image_features = model.encode_image(image)
            processed_image_features = model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            processed_image_features /= processed_image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ processed_image_features.T
        if similarity > threshold:
            indices.append(i)
    return indices


def filtering(args):
    if args.noise_type == 'real':
        dataset_dir = os.path.join('./data/cifar10', args.dataset)
    else:
        dataset_dir = os.path.join('./data', args.dataset, f'{args.noise_type}_{args.noise_rate}')
    train_images = np.load(os.path.join(dataset_dir, 'train', 'train_images.npy'))
    train_processed_images = np.load(os.path.join(dataset_dir, 'train', 'processed_train_images.npy'))
    val_images = np.load(os.path.join(dataset_dir, 'val', 'val_images.npy'))
    val_processed_images = np.load(os.path.join(dataset_dir, 'val', 'processed_val_images.npy'))

    threshold = args.threshold
    train_clean_indices = get_indices(threshold, train_images, train_processed_images)
    val_clean_indices = get_indices(threshold, val_images, val_processed_images)

    return train_clean_indices, val_clean_indices


def filter_clothing1m(args):
    threshold = args.threshold
    train_clean_indices = get_indices_Clothing1M(threshold)

    return train_clean_indices

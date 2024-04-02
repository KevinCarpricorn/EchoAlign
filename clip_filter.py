import numpy as np
from PIL import Image
import open_clip
import torch
from tqdm import tqdm
import os

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


def get_indices(threshold=0.5, images=None, processed_images=None):
    indices = []
    for i in tqdm(range(len(images))):
        image = Image.fromarray(images[i])
        processed_image = Image.fromarray(processed_images[i])
        image = preprocess(image).unsqueeze(0)
        processed_image = preprocess(processed_image).unsqueeze(0)
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
    dataset_dir = os.path.join('./data', args.dataset, f'{args.noise_type}_{args.noise_rate}')
    train_images = np.load(os.path.join(dataset_dir, 'train', 'train_images.npy'))
    train_processed_images = np.load(os.path.join(dataset_dir, 'train', 'processed_train_images.npy'))
    val_images = np.load(os.path.join(dataset_dir, 'val', 'val_images.npy'))
    val_processed_images = np.load(os.path.join(dataset_dir, 'val', 'processed_val_images.npy'))

    threshold = 0.55
    train_clean_indices = get_indices(threshold, train_images, train_processed_images)
    val_clean_indices = get_indices(threshold, val_images, val_processed_images)

    return train_clean_indices, val_clean_indices



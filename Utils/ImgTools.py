import os
import torch
from PIL import Image
import torchvision.transforms as transforms

class ImageLoader:
    """ load the image into np.array """
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '{}/{}'.format(self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


def imagenet_trans_func(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform

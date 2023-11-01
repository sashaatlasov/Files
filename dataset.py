import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Clouds(Dataset):
    def __init__(self, root, indices, transform=None):
        """
        :param root: path to the data folder
        indices: list of indices for train/test samples
        transfrom: transformations of original images
        """
        self.root = root
        self.indices = indices
        self.transform = transform
        self.data_path, self.labels_path = [], []

        self.data_path = []

        for dirname, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename[-4:] == '.jpg':
                      self.data_path += [join(self.root, filename)] 

        self.labels_path =[join(self.root, f + '___fuse.png') for f in self.data_path if isfile(join(self.root, 
                                                                                                     f + '___fuse.png'))]

        self.data_path = np.array(self.data_path)[self.indices]
        self.labels_path = np.array(self.labels_path)[self.indices]

    def __getitem__(self, index):
        """
        :param index: sample index
        :return: tuple (img, target) with the input data and its ground truth
        """
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index])

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.data_path)
    
    
np.random.seed(10) 
train_idx = np.random.choice(np.arange(521), 100)
test_idx = np.random.choice(np.arange(521)[~train_idx], 20)
batch_size = 16 
images_dir ='/Users/sasaatlasov/Desktop/HW1/CloudSkyData/images'

def get_dataloaders():
    train_ds = Clouds(images_dir, train_idx,
                      transform=transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()]))
    test_ds = Clouds(images_dir, test_idx,
                     transform=transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()]))


    train_dataloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=16)
    
    return train_dataloader, test_dataloader
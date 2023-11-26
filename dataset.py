import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from pathlib import Path

class CloudDataset(Dataset):
    '''
    Used https://www.kaggle.com/code/timyapew/fpn-cloud-detection as a basis
    '''
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        
    def combine_files(self, r_file: Path, g_dir, b_dir,nir_dir, gt_dir):
        
        files = {'red': r_file, 
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}

        return files
                                       
    def __len__(self):
        
        return len(self.files)

    def open_as_array(self, idx, invert=False, include_nir=False):
        raw_rgb = np.stack([
            np.array(Image.open(self.files[idx]['red'])),
            np.array(Image.open(self.files[idx]['green'])),
            np.array(Image.open(self.files[idx]['blue'])),
        ], axis=2)

        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), axis=2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        # Normalize
        return raw_rgb / np.iinfo(raw_rgb.dtype).max


    def open_mask(self, idx, add_dims=False):
        
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x.to(torch.float32), y.to(torch.float32)
        

def get_dataloaders(base_dir_train, base_dir_test, batch_size=16):
    
    data = CloudDataset(base_dir_train/'train_red', 
                    base_dir_train/'train_green', 
                    base_dir_train/'train_blue', 
                    base_dir_train/'train_nir',
                    base_dir_train/'train_gt')
    data_test = CloudDataset(base_dir_test/'test_red', 
                    base_dir_test/'test_green', 
                    base_dir_test/'test_blue', 
                    base_dir_test/'test_nir',
                    base_dir_test/'entire_scne_gts')

    train_dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size)
    
    return train_dataloader, test_dataloader
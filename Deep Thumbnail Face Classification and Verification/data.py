import os
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import paths, config

### Classification ###
train_transformations = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(root=paths.train_data_medium,
                                                 transform=train_transformations)

val_dataset = torchvision.datasets.ImageFolder(root=paths.val_data_medium_class,
                                               transform=train_transformations)

class ClassificationTestDataset(Dataset):
    def __init__(self, path):
        self.file_list = open(paths.test_order_class).read().splitlines()
        self.path = path

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.file_list[index]))
        img = torchvision.transforms.ToTensor()(img)
        img = Variable(img)
        img = img.unsqueeze(0)

        return img

### Verification ###
import glob
import re



class VerificationTestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = [i for i in glob.glob(path + '*')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        img = Variable(img)
        idx = int(re.split('/|\.', self.file_list[index])[2])  # get file name as index
        return img, idx

class VerificationTrials(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = open(self.path).read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        idx1 = int(re.split('\.| ', self.file_list[index])[0])  # get file names as index
        idx2 = int(re.split('\.| ', self.file_list[index])[2])  # get file names as index
        return idx1, idx2


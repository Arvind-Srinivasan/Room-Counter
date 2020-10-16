import numpy as np
import torch
import torch.optim as optim
import os
from torchvision import transforms, utils
from PIL import Image

from retinanet import model
# from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
#     Normalizer

from torch.utils.data import Dataset, DataLoader

class FaceMaskDataset(Dataset):

    def __init__(self, imgs_path, anns_path):
        
        valid_img_nums = []
        for file in os.scandir(imgs_path):
             if not file.name.startswith("."):
                    # maksssksksss0.png
                    valid_img_nums.append(file.name[11:-4])

    def __len__(self):


    def __getitem__(self, idx):
        img_path = "maksssksksss" + valid_img_nums[idx] + ".png"
        ann_path = "maksssksksss" + valid_img_nums[idx] + ".xml"
        
        raw_img = Image.open(img_path)
        img = self.transform(crop)


if __name__ == "__main__":
    retinanet = model.resnet18(num_classes=2, pretrained=True)
    print(retinanet)
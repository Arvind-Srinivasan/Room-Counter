import numpy as np
import torch
import torch.optim as optim
import os
from torchvision import transforms, utils
from PIL import Image

import xml.etree.ElementTree as ET
from retinanet import model
# from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
#     Normalizer

from torch.utils.data import Dataset, DataLoader

class_name_to_numbers = {
    'with_mask': 0,
    'without_mask': 1,
    'mask_weared_incorrect': 2
}

class FaceMaskDataset(Dataset):

    def __init__(self, imgs_path, anns_path):
        
        self.valid_img_nums = []
        self.imgs_path = imgs_path
        self.anns_path = anns_path
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        for file in os.scandir(imgs_path):
             if file.name.endswith(".png"):
                    # maksssksksss0.png
                    self.valid_img_nums.append(file.name[12:-4])

    def __len__(self):
        return len(self.valid_img_nums)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, "maksssksksss" + self.valid_img_nums[idx] + ".png")
        ann_path = os.path.join(self.anns_path, "maksssksksss" + self.valid_img_nums[idx] + ".xml")
        
        raw_img = Image.open(img_path)

        img = self.transform(raw_img)

        ann = self.parse_annotation(ann_path)

        return img, ann


    def parse_annotation(self, xmlfile):
        # create element tree object
        tree = ET.parse(xmlfile)

        # get root element
        root = tree.getroot()

        # create empty list for news items
        annotations = []

        # iterate news items
        for item in root.findall('./object'):

            name = item.find('./name')
            name = name.text
            name = class_name_to_numbers[name]

            xmin = item.find('./bndbox/xmin')
            xmin = int(xmin.text)

            ymin = item.find('./bndbox/ymin')
            ymin = int(ymin.text)

            xmax = item.find('./bndbox/xmax')
            xmax = int(xmax.text)

            ymax = item.find('./bndbox/ymax')
            ymax = int(ymax.text)

            annotations.append([ xmin, ymin, xmax, ymax, name])


        # return in form N x (x1, y1, x2, y2, label)
        return torch.tensor(annotations)

def col_func(samples):
    img_list = []
    label_list = []
    for (x, y) in samples:
        img_list.append(x)
        label_list.append(y)
    return torch.stack(img_list, dim=0), label_list


def get_dataloader():
    dataset = FaceMaskDataset(imgs_path="./data/images", anns_path="./data/annotations")
    dl = DataLoader(dataset=dataset, collate_fn=col_func)
    return dl


if __name__ == "__main__":
    # retinanet = model.resnet18(num_classes=2, pretrained=True)
    # print(retinanet)
    dataset = FaceMaskDataset(imgs_path="./data/images", anns_path="./data/annotations")
    print(dataset[345])
import numpy as np
import torch
import torch.optim as optim
import os
from torchvision import transforms, utils
import torchvision.datasets as dset
from PIL import Image
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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

numbers_to_class_names = {
    0: 'with_mask',
    1: 'without_mask',
    2: 'mask_weared_incorrect'
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
        
        raw_img = Image.open(img_path).convert("RGB")

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
        img_list.append(x.to(device))
        label_list.append(y.to(device))
    return torch.stack(img_list, dim=0), label_list


def collater(samples):
    imgs = []
    labels = []
    for (x, y) in samples:
        imgs.append(x)
        labels.append(y)

    max_num_annots = max(annot.shape[0] for annot in labels)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(labels), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(labels):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(labels), 1, 5)) * -1

    return torch.stack(imgs, dim=0), annot_padded




def get_dataloader(params):
    dataset = FaceMaskDataset(imgs_path="./data/images", anns_path="./data/annotations")
    dl = DataLoader(dataset=dataset, collate_fn=collater, **params)
    print("Got Dataloader")
    return dl


VOC_DICT = {
    "__background__": 0,
    "person": 1,
    "bird": 2,
    "cat": 3,
    "cow": 4,
    "dog": 5,
    "horse": 6,
    "sheep": 7,
    "aeroplane": 8,
    "bicycle": 9,
    "boat": 10,
    "bus": 11,
    "car": 12,
    "motorbike": 13,
    "train": 14,
    "bottle": 15,
    "chair": 16,
    "diningtable": 17,
    "pottedplant": 18,
    "sofa": 19,
    "tvmonitor": 20
}

INV_VOC_DICT = {
    0:"__background__",
    1:"person",
    2:"bird",
    3:"cat",
    4:"cow",
    5:"dog",
    6:"horse",
    7:"sheep",
    8:"aeroplane",
    9:"bicycle",
    10:"boat",
    11:"bus",
    12:"car",
    13:"motorbike",
    14:"train",
    15:"bottle",
    16:"chair",
    17:"diningtable",
    18:"pottedplant",
    19:"sofa",
    20:"tvmonitor"
}

def get_voc_generator(path, year, image_set, batch_size, shuffle=True):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    def detection_collate_fn(sample_list):
        img_batch = []
        target_batch = []
        max_len = 0
        for (x, y) in sample_list:
            x_scale = 256 / x.size[0]
            y_scale = 256 / x.size[1]
            img_batch.append(transform(x))
            y = torch.stack([y[:, 0] * x_scale, y[:, 1] * y_scale, y[:, 2] * x_scale,y[:, 3] * y_scale, y[:, 4]], dim=1)
            if y.shape[0] > max_len:
                max_len = y.shape[0]
            target_batch.append(y)
            
        target_tensor = torch.ones((len(target_batch), max_len, 5)) * -1
        for i in range(target_tensor.shape[0]):
            j = target_batch[i].shape[0]
            target_tensor[i, :j] = target_batch[i]
        img_batch = torch.stack(img_batch)
        target_tensor = target_tensor
        return img_batch, target_tensor
    
    voc_data = dset.VOCDetection(root = path,
                                  year = year,
                                  image_set = image_set,
                                  target_transform=voc_target_transform)
    voc_generator = DataLoader(voc_data, batch_size=batch_size, collate_fn=detection_collate_fn, shuffle=shuffle, drop_last=True)
    return voc_generator

def voc_target_transform(y):
    truths = []
    for elem in y["annotation"]["object"]:
        truth = torch.zeros((5, ), dtype=torch.float)
        truth[0] = int(elem["bndbox"]["xmin"])
        truth[1] = int(elem["bndbox"]["ymin"])
        truth[2] = int(elem["bndbox"]["xmax"])
        truth[3] = int(elem["bndbox"]["ymax"])
        truth[4] = VOC_DICT[elem["name"]] - 1
        truths.append(truth)
    if truths:
        truth_array = torch.stack(truths, dim=0)
    else:
        truth_array = torch.zeros((0, 5), dtype=torch.float)

    return truth_array


if __name__ == "__main__":
    # retinanet = model.resnet18(num_classes=2, pretrained=True)
    # print(retinanet)
#     dataset = FaceMaskDataset(imgs_path="./data/images", anns_path="./data/annotations")
#     print(dataset[345])
    
    voc_train_generator = get_voc_generator('/srv/datasets/pascalvoc-2012', "2012", "train", 3)
    print(next(iter(voc_train_generator))[1])
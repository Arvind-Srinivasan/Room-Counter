import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from retinanet import model

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer

from data import get_dataloader, numbers_to_class_names

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

WEIGHTS = "weights/0_weights.pt"
# Parameters
params = {'batch_size': 1,
          'num_workers': 6}

eval_generator = get_dataloader(params)

retinanet = model.resnet18(num_classes=3, pretrained=True).to(device)
retinanet.load_state_dict(torch.load(WEIGHTS))

use_gpu = True

if use_gpu:
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

retinanet.eval()


import torch.nn.functional as F

SIZE = 256

def infer(image):

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)
    image = F.interpolate(image, size=256)

    with torch.no_grad():

        if torch.cuda.is_available():
            scores, classification, transformed_anchors = retinanet(image.cuda())
        else:
            scores, classification, transformed_anchors = retinanet(image)

        idxs = np.where(scores.cpu() > 0.5)

        return_list = list()
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0]) / SIZE
            y1 = int(bbox[1]) / SIZE
            x2 = int(bbox[2]) / SIZE
            y2 = int(bbox[3]) / SIZE
            label_name = numbers_to_class_names[int(classification[idxs[0][j]])]

            return_list.append((x1, y1, x2, y2, label_name))

        return return_list

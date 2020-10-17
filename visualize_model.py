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

# Parameters
params = {'batch_size': 1,
          'num_workers': 6}

eval_generator = get_dataloader(params)


def visualize(weights):
    retinanet = model.resnet18(num_classes=3, pretrained=True).to(device)
    retinanet.load_state_dict(torch.load(weights))


    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


    for idx, (local_batch, local_labels) in enumerate(eval_generator):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(local_batch.cuda())
            else:
                scores, classification, transformed_anchors = retinanet(local_batch)
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(local_batch[0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = numbers_to_class_names[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)

            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    visualize("weights/0_weights.pt")
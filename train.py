import argparse, os, glob, sys, json, random, tqdm
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import math

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.append('../vision/references/detection'.replace('/', os.sep))
import engine, utils
import transforms as T
import common
import dataset
import model


def trainer(train, model, optimizer):
    print("---------- Start Training ----------")
    
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    try:
        with tqdm(trainloader, ncols=100) as pbar:
            train_loss = 0.0
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss += loss_value
        return train_loss
    except ValueError:
        pass


def validater(valid, model):
    print("---------- Start Validating ----------")
    
    validloader = torch.utils.data.DataLoader(
        valid, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    try:
        with tqdm(validloader, ncols=100) as pbar:
            valid_loss = 0.0
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                valid_loss += loss_value

        return valid_loss
    except ValueError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()

    # open via json file
    tmp = open(common.VIA_JSON, 'r')
    annotation = json.load(tmp)
    tmp.close()

    # setup model and device
    model = model.get_model_instance_segmentation(common.NUM_CLASSES)
    model, device = common.setup_device(model)

    # dataset
    img_paths = glob.glob(common.TRAIN_DATA_IMGS)
    dataset = dataset.Dataset(img_paths, annotation)
    train_size = int(len(dataset) * 0.9)
    print(train_size)
    valid_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])
#         train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn)
#         valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=common.LEARNING_RATE)

    # Training
    early_stopping = [np.inf, 3, 0]
    for epoch in range(common.NUM_EPOCHS):
        train_loss = trainer(train, model, optimizer)
        with torch.no_grad():
            valid_loss = validater(valid, model)
#             metric_logger = engine.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=30, is_train=True)
#             loss = metric_logger.__getattr__('loss').median
# 
#             valid_metric_logger = engine.train_one_epoch(model, optimizer, valid_loader, device, epoch, print_freq=30, is_train=False)
#             valid_loss = valid_metric_logger.__getattr__('loss').median

        # early stopping
        if valid_loss < early_stopping[0]:
            early_stopping[0] = valid_loss
            early_stopping[-1] = 0
            torch.save(model.state_dict(), "model.pth")
            print(early_stopping)
        else:
            early_stopping[-1] += 1
            if early_stopping[-1] == early_stopping[1]:
                break

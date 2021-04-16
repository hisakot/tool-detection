import cv2
import numpy as np
import os
import random
import torch


def horizontal_flip(img, masks, boxes, p):
    if random.random() < p:
        img = img[:,::-1,:]
        for idx in range(masks.shape[0]):
            masks[idx] = masks[idx,:,::-1]
        boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]

    return img, masks, boxes

class Dataset(object):
    def __init__(self, img_paths, annotation):
        self.img_paths = img_paths
        tmp = {}
        for k, v in annotation.items():
            tmp[v["filename"]] = v
        self.annotation = tmp

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, 1)

        regions = self.annotation[img_path.split(os.sep)[-1]]["regions"]
        num_objs = len(regions)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        masks = np.zeros((num_objs, img.shape[0], img.shape[1]), dtype=np.uint8)
        labels = np.zeros(num_objs, dtype=np.int64)

        for idx, region in enumerate(regions):
            tmp = region['shape_attributes']
            xs = tmp['all_points_x']
            ys = tmp['all_points_y']
            # bbox
            boxes[idx] = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            # mask
            vertex = [[x, y] for x, y in zip(xs, ys)]
            cv2.fillPoly(masks[idx], [np.array(vertex)], 1)
            # label
            labels[idx] = list(region['region_attributes']['tool'].keys())[0]
            # horizontal
            img, masks, boxes = horizontal_flip(img, masks, boxes, 0.5)

        img = img / 255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))
        boxes = torch.from_numpy(boxes)

        target = {
            "boxes": boxes,
            "masks": torch.from_numpy(masks),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2]-boxes[:, 0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        return img, target

    def __len__(self):
        return len(self.img_paths)


import cv2
import numpy as np
import os

import torch

DATASET_CACHE = "./dataset_cache"

class Dataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "org_imgs"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images[idx])
        mask_path = os.path.join(self.root, self.masks[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask= cv2.imread(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = list()
        for i in range(num_objs):
            pos = np.where(masks[i])
            x1 = np.min(pos[1])
            x2 = np.max(pos[1])
            y1 = np.min(pos[0])
            y2 = np.max(pos[0])
            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

def setup_data():
    datas = Dataset("../datas/green_gloves/", None)

    try:
        cache = torch.load(DATASET_CACHE)
        datas.dataset = cache["dataset"]
        datas.length = cache["length"]

    except FileNotFoundError:
        dataset_dicts = make_dataset()
        datas.dataset = dataset_dicts
        datas.length = len(datas.dataset)

        cache_dict = {"dataset" : datas.dataset,
                      "length" : datas.length,}
        torch.save(cache_dict, DATASET_CACHE)

    return datas


import ast
import cv2
import numpy as np
import os
import pandas as pd

import torch

DATASET_CACHE = "./dataset_cache"
DATA_CSV = "../datas/tool_data/annotations.csv"
ORG_IMG_DIR = "../datas/tool_data/org_imgs/"

class Dataset(object):
    def __init__(self):
        self.dataset = list()
        self.length = 0

    def __getitem__(self, idx):
        # load image
        image_path = self.dataset[idx]["image_path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)) # (c, h, w)
        image = torch.tensor(image, dtype=torch.int64) # tensor

        # load target
        target = dict()
        target["polygons"] = self.dataset[idx]["polygons"]
        target["tool_labels"] = self.dataset[idx]["tool_labels"]
        target = torch.tensor(target, dtype=torch.int64)

        return image, target

    def __len__(self):
        return len(self.dataset)

def make_dataset():
    dataset_dicts = list()

    polygons = list()
    tool_labels = list()
    csv_data = pd.read_csv(DATA_CSV, usecols=["filename",
                                              "region_shape_attributes",
                                              "region_attributes"])
    # csv_data.values[i] ->
    # ['*.png', '{"name":"polygon", "x":[1, 2, ...], "y":[1, 2, ...]}', '{"tool":{"n":true}}']
    for i in range(len(csv_data.values) - 1):
        if csv_data.values[i][0] == csv_data.values[i + 1][0]:
            polygons.append(ast.literal_eval(csv_data.values[i][1]))
            tool_labels.append(csv_data.values[i][2])
        else:
            polygons.append(ast.literal_eval(csv_data.values[i][1]))
            tool_labels.append(csv_data.values[i][2])
            image_path = ORG_IMG_DIR + csv_data.values[i][0]
            dataset_dicts.append({"image_path" : image_path,
                                  "polygons" : polygons,
                                  "tool_labels" : tool_labels})
            polygons = list()
            tool_labels = list()

    return dataset_dicts

def setup_data():
    datas = Dataset()

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

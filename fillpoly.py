
import ast
import cv2
import numpy as np
import pandas as pd
import random
import re

import torch

DATA_CSV = "../data/tool/annotations.csv"
DATASET_CACHE = "./dataset_cache"

polygons = list()
tool_labels = list()
csv_data = pd.read_csv(DATA_CSV, usecols=["filename",
                                          "region_count",
                                          "region_id",
                                          "region_shape_attributes",
                                          "region_attributes"])
# csv_data.values[i] ->
# ['*.png', '{"name":"polygon", "x":[1, 2, ...], "y":[1, 2, ...]}', '{"tool":{"n":true}}']

dataset = list()
box_list = list()
tool_label = list()
img = np.zeros((1080, 1920, 3), np.uint8)
for i in range(len(csv_data.values) - 1):
    region_shape_attributes = ast.literal_eval(csv_data.values[i][3])
    xy_list = list()
    for j, x in enumerate(region_shape_attributes["all_points_x"]):
        xy_list.append((x, region_shape_attributes["all_points_y"][j]))
    vertex = np.array(xy_list)
    b = random.randrange(256)
    g = random.randrange(256)
    r = random.randrange(256)
    cv2.fillPoly(img, [vertex], (b, g, r))
    box_list.append([min(region_shape_attributes["all_points_x"]),
                     min(region_shape_attributes["all_points_y"]),
                     max(region_shape_attributes["all_points_x"]),
                     max(region_shape_attributes["all_points_y"])])
    tool_label.append(int(re.sub('[^0-9]+', '', csv_data.values[i][4])))

    if csv_data.values[i][1] - 1 == csv_data.values[i][2]:
        cv2.imwrite("../data/tool/masks/" + csv_data.values[i][0], img)
        dataset.append({"filename" : csv_data.values[i][0],
                        "box_list" : box_list,
                        "tool_label" : tool_label,})
        box_list = list()
        tool_label = list()
        img = np.zeros((1080, 1920, 3), np.uint8)


length = len(dataset)

cache_dict = {"dataset" : dataset,
              "length" : length,}
torch.save(cache_dict, DATASET_CACHE)

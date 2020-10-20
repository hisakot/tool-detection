
import ast
import cv2
import numpy as np
import pandas as pd
import random

DATA_CSV = "../data/tool/annotations.csv"


polygons = list()
tool_labels = list()
csv_data = pd.read_csv(DATA_CSV, usecols=["filename",
                                          "region_count",
                                          "region_id",
                                          "region_shape_attributes",
                                          "region_attributes"])
# csv_data.values[i] ->
# ['*.png', '{"name":"polygon", "x":[1, 2, ...], "y":[1, 2, ...]}', '{"tool":{"n":true}}']

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
    if csv_data.values[i][1] - 1 > csv_data.values[i][2]:
        cv2.fillPoly(img, [vertex], (b, g, r))
    else:
#         img = cv2.resize(img, (960, 540))
#         cv2.imshow("img", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        cv2.imwrite("../data/tool/masks/" + csv_data.values[i][0], img)
        img = np.zeros((1080, 1920, 3), np.uint8)


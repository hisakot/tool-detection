import cv2
import glob
import numpy as np
from PIL import Image
import torch
import torchvision
import tqdm

import transforms as T
import common
import model

def get_coloured_mask(mask, pred_cls):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    colours = [[0, 0, 0],[0, 255, 0],[0, 255, 255],[255, 255, 0],[80, 70, 180],[180, 40, 250],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[common.CLASS_NAMES.index(pred_cls)]
    coloured_mask = np.stack([r, g, b], axis=2)

    return coloured_mask

if __name__ == '__main__':
    # load images
    img_paths = glob.glob(common.TRAIN_DATA_IMGS)
    # load model
    model = model.get_model_instance_segmentation(common.NUM_CLASSES)
    model, device = common.setup_device(model)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    confidence = 0.5
    for idx in range(len(img_paths)):
        # Prediction
        img_path = img_paths[idx]
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = torchvision.transforms.functional.to_tensor(img)
        pred = model([img.to(device)])

        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        if masks.ndim == 2: masks = masks.reshape([1, masks.shape[0], masks.shape[1]])
        pred_class = [common.CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        if len(pred_t) == 0:
            masks = []
            boxes = []
            pred_cls = []
        else:
            pred_t = pred_t[-1]
            masks = masks[:pred_t+1]
            boxes = pred_boxes[:pred_t+1]
            pred_cls = pred_class[:pred_t+1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        multi_tool_masks = np.zeros((img.shape[0], img.shape[1], common.NUM_CLASSES))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for i in range(len(masks)):
            rgb_mask = get_coloured_mask(masks[i], pred_cls[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.9, 0)
            for j, class_name in enumerate(common.CLASS_NAMES):
                if pred_cls[i] == class_name:
                    multi_tool_masks[:, :, j] = masks[i]
        # np.save(common.SAVE_NPY_DIR + str(idx+1).zfill(6), cv2.resize(multi_tool_masks, (320, 180)))
        cv2.imwrite(common.SAVE_DIR + str(idx+1).zfill(6) + ".png", img)

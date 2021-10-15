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

def horizontal_flip(img, masks, boxes, p):
    if random.random() < p:
        img = img[:,::-1,:]
        for idx in range(masks.shape[0]):
            masks[idx] = masks[idx,:,::-1]
        boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]

    return img, masks, boxes

class Dataset(object):
    def __init__(self, img_paths, annotation, is_train):
        self.img_paths = img_paths
        tmp = {}
        for k, v in annotation.items():
            tmp[v["filename"]] = v
        self.annotation = tmp
        self.is_train = is_train

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv.imread(img_path, 1)

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
            cv.fillPoly(masks[idx], [np.array(vertex)], 1)
            # label
            labels[idx] = list(region['region_attributes']['tool'].keys())[0]

        if self.is_train:
            img, masks, boxes = horizontal_flip(img, masks, boxes, 0.5)

        img = img / 255.
        img = img.transpose(2,0,1)
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

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

def trainer(train, model, optimizer):
    print("---------- Start Training ----------")
    
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

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

def tester(test, model):
    print("---------- Start Testing ----------")
    
    testloader = torch.utils.data.DataLoader(
        test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    try:
        with tqdm(testloader, ncols=100) as pbar:
            test_loss = 0.0
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

                test_loss += loss_value

        return test_loss
    except ValueError:
        pass


if __name__ == '__main__':
    CLASS_NAMES = ['background', 'assistant', 'main']
    NUM_CLASSES = len(CLASS_NAMES)
    NUM_EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--evaluation', action='store_true')
    args = parser.parse_args()

    # device
    device = torch.device('cuda')

    # open via json file
    # tmp = open('via_region_data.json', 'r')
    tmp = open('annotation_data_main_or_not.json', 'r')
    annotation = json.load(tmp)
    tmp.close()

    # model
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)
    model = torch.nn.DataParallel(model)

    img_paths = glob.glob("../data/tool2/annotation_img/*.png")
    if not args.evaluation:
        # dataset
        dataset = Dataset(img_paths, annotation, is_train=True)
        train_size = int(len(dataset) * 0.9)
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
#         train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn)
#         test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn)

        # optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

        # Training
        early_stopping = [np.inf, 3, 0]
        for epoch in range(NUM_EPOCHS):
            train_loss = trainer(train, model, optimizer)
            with torch.no_grad():
                test_loss = tester(test, model)
#             metric_logger = engine.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=30, is_train=True)
#             loss = metric_logger.__getattr__('loss').median
# 
#             test_metric_logger = engine.train_one_epoch(model, optimizer, test_loader, device, epoch, print_freq=30, is_train=False)
#             test_loss = test_metric_logger.__getattr__('loss').median

            # early stopping
            if test_loss < early_stopping[0]:
                early_stopping[0] = test_loss
                early_stopping[-1] = 0
                torch.save(model.state_dict(), "model.pth")
                print(early_stopping)
            else:
                early_stopping[-1] += 1
                if early_stopping[-1] == early_stopping[1]:
                    break

    else:
        def get_coloured_mask(mask, pred_cls):
            # colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
            r = np.zeros_like(mask).astype(np.uint8)
            g = np.zeros_like(mask).astype(np.uint8)
            b = np.zeros_like(mask).astype(np.uint8)
            # r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
            colours = [[0, 0, 0],[0, 255, 0],[0, 255, 255],[255, 255, 0],[80, 70, 180],[180, 40, 250],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
            r[mask == 1], g[mask == 1], b[mask == 1] = colours[CLASS_NAMES.index(pred_cls)]
            coloured_mask = np.stack([r, g, b], axis=2)
            return coloured_mask

        img_paths = glob.glob("../main20170707/org_imgs/*.png")
        # img_paths = glob.glob("../data/tool/org_imgs/*.png")
        # dataset = Dataset(img_paths, annotation, is_train=False)
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
        confidence = 0.5
        # for idx in tqdm.tqdm(range(dataset.__len__())):
        for idx in tqdm(range(len(img_paths))):
            # Prediction
            # img, _ = dataset.__getitem__(idx)

            img_path = img_paths[idx]
            img = Image.open(img_path)
            transform = T.Compose([T.ToTensor()])
            img = torchvision.transforms.functional.to_tensor(img)
            pred = model([img.to(device)])

            pred_score = list(pred[0]['scores'].detach().cpu().numpy())
            pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
            masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
            if masks.ndim == 2: masks = masks.reshape([1, masks.shape[0], masks.shape[1]])
            pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
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

            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            multi_tool_masks = np.zeros((img.shape[0], img.shape[1], NUM_CLASSES))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            for i in range(len(masks)):
                rgb_mask = get_coloured_mask(masks[i], pred_cls[i])
                img = cv.addWeighted(img, 1, rgb_mask, 0.7, 0)
                # if len(boxes) != 0:
                #     cv.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=3)
                #     cv.putText(img, pred_cls[i], boxes[i][0], cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), thickness=2)
                # for j, class_name in enumerate(CLASS_NAMES):
                #     if pred_cls[i] == class_name:
                #         multi_tool_masks[:, :, j] = masks[i]
            # np.save('../main20170707/multi_channel_tool/'+str(idx+43383).zfill(6), cv.resize(multi_tool_masks, (320, 180)))
            img = cv.resize(img, (960, 540))
            cv.imwrite('../main20170707/result_imgs/multi_tools_imgs/'+img_paths[idx].split(os.sep)[-1], img)

import argparse
import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import random

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T
import utils
# from engine import train_one_epoch, evaluate

DATASET_CACHE = "./dataset_cache"
MODEL_SAVE_PATH = "./models/tool_detection/"
INF_IMGS_PATH = "../main20200214/org_imgs/"
# INF_IMGS_PATH = "../data/tool2/org_imgs/"

class Dataset(object):
    def __init__(self, root, transforms, dataset, length):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "org_imgs"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

        self.dataset = dataset
        self.length = length

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "org_imgs", self.dataset[idx]["filename"])
        mask_path = os.path.join(self.root, "masks", self.dataset[idx]["filename"])
        box_list = self.dataset[idx]["box_list"]
        tool_label = self.dataset[idx]["tool_label"]

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        pass
        # transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def trainer(train, model, optimizer, lossfunc):
    print("---------- Start Training ----------")
    
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    try:
        with tqdm(trainloader, ncols=100) as pbar:
            train_loss = 0.0
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#                 images, labels = Variable(images), Variable(targets)
#                 images, labels = images.to(device), targets.to(device)

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
        test, batch_size=4, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    try:
        with tqdm(testloader, ncols=100) as pbar:
            test_loss = 0.0
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#                 images, labels = Variable(images), Variable(targets)
#                 images, labels = images.to(device), targets.to(device)

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

def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    # r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[3]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = torchvision.transforms.functional.to_tensor(img)

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    if len(pred_t) == 0:
        masks = (pred[0]['masks']>confidence).squeeze().detach().cpu().numpy()
        if masks.shape == (540, 960):
            masks = masks[np.newaxis, :, :]
            print(masks.shape)
        pred_boxes = []
        pred_class = []
        return masks, pred_boxes, pred_class
    pred_t = pred_t[-1]
    # pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>confidence).squeeze().detach().cpu().numpy()
    if masks.shape == (540, 960):
        masks = masks[np.newaxis, :, :]
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def segment_instance(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, confidence)
    img = cv2.imread(img_path)
    mask_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_save_path = img_path.replace('org_imgs', 'tool_masks')
    for i in range(len(masks)):
        if len(masks) == 0 or masks[i].ndim != 2:
            mask_img = np.zeros((320, 180))
            break
        rgb_mask = get_coloured_mask(masks[i])
        mask_img = cv2.addWeighted(mask_img, 1, rgb_mask, 1, 0)

    mask_img = cv2.resize(mask_img, (320, 180))
    cv2.imwrite(mask_save_path, mask_img)
        
            # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
            # cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    # save_path = img_path.replace('org_imgs', 'tool_detected')
    # img = cv2.resize(img, (320, 180))
    # cv2.imwrite(save_path, img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--evaluation', default=False, help='evaluation mode')
    args = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
#    dataset_cache = torch.load(DATASET_CACHE)
#    dataset = dataset_cache["dataset"]
#    length = dataset_cache["length"]
#    data = Dataset('../data/tool2/', get_transform(train=True), dataset, length)
#
#    # split the dataset in train and test set
#    train_size = int(length * 0.8)
#    test_size = length - train_size
#    train, test = torch.utils.data.random_split(data, [train_size, test_size])

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # set to evaluation mode
    if args.evaluation:
        save_model = glob.glob(MODEL_SAVE_PATH + "*")[0]
        print(save_model)
        checkpoint = torch.load(save_model, map_location=device)
        if torch.cuda.device_count() >= 1:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                name = k[7:] # remove "module."
                state_dict[name] = v
            model.load_state_dict(state_dict)
        model.eval()
        CLASS_NAMES = ['__background__', 'tool']
        model.to(device)

        # inf_img = INF_IMGS_PATH + "000040.png"
        # segment_instance(inf_img, confidence=0.7)
        # exit()
        inf_imgs = glob.glob(INF_IMGS_PATH + '*')
        for inf_img in inf_imgs:
            segment_instance(inf_img, confidence=0.7)
        exit()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    lossfunc = torch.nn.MSELoss

    # tensorboard
    writer = SummaryWriter(log_dir="./logs")

    loss_list = list()
    for epoch in range(100):
        try:
            # train
            train_loss = trainer(train, model, optimizer, lossfunc)
            loss_list.append(train_loss)

            # update the learning rate
            lr_scheduler.step()

            # test
            with torch.no_grad():
                test_loss = tester(test, model)

            print("%d : train_loss : %.3f" % (epoch + 1, train_loss))
            print("%d : test_loss : %.3f" % (epoch + 1, test_loss))

            # save model
            torch.save({
                "epoch" : epoch + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : loss_list,
                }, "models/" + str(epoch + 1))

            # tensorboard
            writer.add_scalar("Train_Loss", train_loss, epoch)
            writer.add_scalar("Test_Loss", test_loss, epoch)

        except ValueError:
            continue



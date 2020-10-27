import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import math

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T
from engine import train_one_epoch, evaluate
import utils

DATASET_CACHE = "./dataset_cache"

class Dataset(object):
    def __init__(self, root, transforms, dataset, length):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "org_imgs"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

        self.dataset = dataset
        self.length = length

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "org_imgs", self.dataset[idx]["filename"])
        mask_path = os.path.join(self.root, "masks", self.dataset[idx]["filename"])
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
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def trainer(train, model, optimizer, lossfunc):
    print("---------- Start Training ----------")
    
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)

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
    print("---------- Start Training ----------")
    
    testloader = torch.utils.data.DataLoader(
        test, batch_size=1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

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

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset_cache = torch.load(DATASET_CACHE)
    dataset = dataset_cache["dataset"]
    length = dataset_cache["length"]
    data = Dataset('../data/tool/', get_transform(train=True), dataset, length)

    # split the dataset in train and test set
    train_size = int(length * 0.8)
    test_size = length - train_size
    train, test = torch.utils.data.random_split(data, [train_size, test_size])

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

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
            writer.add_scalar("Train_Loss", train_loss, epoch)

        except ValueError:
            continue

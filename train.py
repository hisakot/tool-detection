import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

import common
import dataset
import model


def trainer(train, model, optimizer, lossfunc):
    print("---------- Start Training ----------")
    
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=2, shuffle=True, num_workers=4)

    try:
        with tqdm(trainloader, ncols=100) as pbar:
            train_loss = 0.0
            for images, targets in pbar:
                images, labels = Variable(images), Variable(targets)
                images, labels = images.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = lossfunc(outputs, labels)
                loss.background()
                optimizer.step()
                train_loss += loss.item()
        return train_loss
    except ValueError:
        pass

def tester(test, model):
    print("---------- Start Training ----------")
    
    testloader = torch.utils.data.DataLoader(
        test, batch_size=2, shuffle=False, num_workers=4)

    try:
        with tqdm(testloader, ncols=100) as pbar:
            test_loss = 0.0
            for images, targets in pbar:
                images, labels = Variable(images), Variable(targets)
                images, labels = images.to(device), targets.to(device)

                outputs = model(images)
                loss = lossfunc(outputs, labels)
                test_loss += loss.item()
        return test_loss
    except ValueError:
        pass

if __name__ == '__main__':
    # our dataset has two classes only - background and tool
    num_classes = 9
    
    # load dataset
    print("--------- Loading Data ----------")
    datas = dataset.setup_data()
    print("--------- Finished Loading Data ----------")

    # split train and test
    train_size = int(round(datas.length * 0.8))
    test_size = datas.length - train_size
    train, test = torch.utils.data.random_split(datas, [train_size, test_size])
 
 
    # get the model using our helper function
    model = model.mask_rcnn(num_classes)
    model, device = common.setup_device(model)
 
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lossfunc = torch.nn.MSELoss
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
 
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

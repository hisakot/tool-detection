import torch
import torch.nn as nn

def setup_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use ", torch.cuda.device_count(), "GPUs ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.to(device)

    return model, device


import torch
import torch.nn as nn

# CLASS_NAMES = ['background', 'tool']
CLASS_NAMES = ['background', 'main', 'assistant']
# CLASS_NAMES = ['background', 'forceps', 'tweezers', 'electrical-scalpel',
#                'scalpels', 'hook', 'syringe', 'needle-holder', 'pen']
NUM_CLASSES = len(CLASS_NAMES)
NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
VIA_JSON = 'annotation_data_main_or_not.json'
TRAIN_DATA_IMGS = "../data/tool/annotation_imgs/*.png"
TEST_DATA_IMGS = "../main20170707/org_imgs/*.png"
SAVE_NPY_DIR = "../main20170707/multi_channel_tool/"
SAVE_DIR = "../data/tool/masked/"

def setup_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use ", torch.cuda.device_count(), "GPUs ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.to(device)

    return model, device

import numpy as np
import os
import torch
from torch import nn
import time
import sys
import cv2
from torchvision import models
from torch import optim
# from torch.utils.data import *
from torch.utils.data import BatchSampler
import torch.nn.functional as F
from torch.autograd import Variable
from config import get_cfg_defaults
from models.model import create_model
from utils.visualize import visualize

def preloader():
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    config_path=f'{current_path}/config/hrnet_plate.yaml'
    seg_weights=f'{current_path}/weights/hrnetv2_hrnet_plate_199.pth'
    output_dir=f'{current_path}/'
    data_name='plate'
    
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(
        [
            "train.config_path",
            config_path,
            "train.output_dir",
            output_dir,
            "dataset.data_name",
            data_name,
        ]
    )
    print(torch.load(os.path.join(f"{current_path}/data", data_name + ".pth"),map_location=torch.device('cpu')))

    model = create_model(cfg)
    # if torch.cuda.is_available():
    #     model.cuda()
    model = nn.DataParallel(model)
    print(torch.load(seg_weights,map_location=torch.device('cpu')).keys())
    model.load_state_dict(torch.load(seg_weights,map_location=torch.device('cpu'))["state_dict"])
    model.eval()
    return model,cfg,
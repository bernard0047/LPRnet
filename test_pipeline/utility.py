import numpy as np
import torch
import time
import os
import sys
import cv2
from lprnet import *
from dataset.augmentations import *
from func import *

def runner(frame,model,cfg):

    current_path = os.path.dirname(os.path.abspath(__file__))
    
    lpr_weights=f'{current_path}/weights/iter2.pth'
    debug_program=True
    cuda=False

    frame = cv2.resize(frame, (1920, 1080))
    frame1 = frame
    image = preprocess_image(frame, cfg)
    # if torch.cuda.is_available():
    #     image = image.cuda()

    with torch.no_grad():
        prediction = model(image, (cfg.dataset.height, cfg.dataset.width))
        prediction = (
            torch.argmax(prediction["output"][0], dim=1)
            .cpu()
            .squeeze(dim=0)
            .numpy()
            .astype(np.uint8)
        ).reshape(frame.shape[0], frame.shape[1], 1)

        cropped_images, coordinates, centroid = plate_cropper(prediction, frame)
        final_image = frame1
        if len(cropped_images) != 0:
            labels = get_lprnet_preds(cropped_images, lpr_weights,cuda)
        data_dictionary = {}
        if debug_program:
            if len(cropped_images) != 0:
                data_dictionary = details(prediction, labels, coordinates, centroid)
                final_image = overlay_colour(prediction, frame, centroid)
                final_image = write_string(
                    prediction, frame, coordinates, centroid, labels
                )

        return final_image, data_dictionary

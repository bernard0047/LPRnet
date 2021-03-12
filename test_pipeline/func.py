import numpy as np
import os
import sys
import cv2
from config import get_cfg_defaults
from models.model import create_model
from utils.visualize import visualize
from tqdm import tqdm
from dataset.augmentations import *

classes = {
    10: [(102, 102, 0), "auto front"],
    9: [(104, 104, 104), "auto back"],
    8: [(255, 102, 102), "bus front"],
    7: [(255, 255, 0), "bus back"],
    5: [(255, 0, 127), "truck back"],
    6: [(204, 0, 204), "truck front"],
    4: [(102, 204, 0), "bike front"],
    2: [(0, 0, 255), "car front"],
    3: [(0, 255, 0), "bike back"],
    1: [(255, 0, 0), "car back"],
    0: [(0, 0, 0), "background"],
}

def plate_cropper(image, imagergb):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    size_factor = 1.03
    cropped_images = []
    coordinates = []
    centroid = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400:
            continue

        temp_rect = []
        rect = cv2.minAreaRect(c)
        centroid.append(rect)
        temp_rect.append(rect[0][0])
        temp_rect.append(rect[0][1])
        temp_rect.append(rect[1][0] * size_factor)
        temp_rect.append(rect[1][1] * size_factor)
        temp_rect.append(rect[2])
        rect = (
            (temp_rect[0], temp_rect[1]),
            (temp_rect[2], temp_rect[3]),
            temp_rect[4],
        )

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        coordinates.append(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(imagergb, M, (width, height))
        if width < height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cropped_images.append(warped)
        # cv2.imwrite('/home/sanchit/Desktop/warped'+'/'+str(i)+str(width)+'.jpg',warped)
        # i=i+1

    return cropped_images, coordinates, centroid

def overlay_colour(prediction, frame, centroid):

    temp_img = frame.copy()
    for i in range(len(centroid)):
        temp = centroid[i]
        pred_class = prediction[int(temp[0][1])][int(temp[0][0])][0]
        box = cv2.boxPoints(temp)
        box = np.int0(box)
        cv2.drawContours(temp_img, [box], 0, classes[pred_class][0], -1)
    cv2.addWeighted(temp_img, 0.5, frame, 0.5, 0, frame)
    return frame


def write_string(prediction, image, coordinates, centroid, labels):
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2

    for i in range(len(coordinates)):
        temp = centroid[i]
        box = cv2.boxPoints(temp)
        box = np.int0(box)
        pred_class = prediction[int(temp[0][1])][int(temp[0][0])]
        pred_class = pred_class[0]
        
        xmin, ymin, xmax, ymax=sys.maxsize,sys.maxsize,0,0
        
        for j in range(len(coordinates[i])):
            if(coordinates[i][j][0]<xmin):
                xmin=coordinates[i][j][0]
            if(coordinates[i][j][1]<ymin):
                ymin=coordinates[i][j][1]
            if(coordinates[i][j][0]>xmax):
                xmax=coordinates[i][j][0]
            if(coordinates[i][j][1]>ymax):
                ymax=coordinates[i][j][1]
                
        xmin=xmin-(xmax-xmin)//15
        xmax=xmax+(xmax-xmin)//15
        ymin=ymin-(ymax-ymin)//15
        ymax=ymax+(ymax-ymin)//15
        ymax=ymax+(ymax-ymin)//15
                    
        # cv2.drawContours(image, [coordinates[i]], 0, (0, 255, 0), 3)
        image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax),(0, 255, 0) , 3)
        org = (
            int(temp[0][0] - temp[1][0] // 2),
            int(temp[0][1] - temp[1][1] // 2),
        )
        del temp
        image[org[1] - 25 : org[1], org[0] : org[0] + 25, :] = (255, 255, 255)
        image = cv2.putText(
            image,
            str(i+1),
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return image

def details(prediction, labels, coordinates, centroid):
    data_dictionary = {}
    for i in range(len(coordinates)):
        temp = centroid[i]
        pred_class = prediction[int(temp[0][1])][int(temp[0][0])][0]
        print(pred_class)
        box = cv2.boxPoints(temp)
        box = np.int0(box)
        data_dictionary[i] = [
            coordinates[i],
            classes[pred_class][1].split()[0],
            classes[pred_class][1].split()[1],
            labels[i],
        ]

    return data_dictionary


def preprocess_image(image, cfg):
    image = normalize(image, cfg)
    return torch.unsqueeze(image, dim=0)

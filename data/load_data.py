from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
#aaryan
CHARS = [
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        print("1dir found, size: ",len(self.img_paths))
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        try:
            Image = cv2.imread(filename)
            height, width, _ = Image.shape
            Image = cv2.resize(Image, self.img_size)
            Image = self.PreprocFun(Image)
            basename = os.path.basename(filename)
            imgname, suffix = os.path.splitext(basename)
            imgname = ''.join(e for e in imgname if e.isalnum())
            label = list()
            for c in imgname:
                c = c.upper()
                # one_hot_base = np.zeros(len(CHARS))
                # one_hot_base[CHARS_DICT[c]] = 1
                label.append(CHARS_DICT[c])
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

            return Image, label, len(label)
        except:
            print("Error in image: ",filename)
            print("deleting: ")
            #os.remove(filename)
            
    
    def transform(self, img):
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img -= 127.5
        img *= 0.0078125
        #thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        img = np.reshape(img, img.shape + (1,))
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        if len(label)<4:
            print("Error label, Please check!")
            return False
        else:
            return True


'''
created:31-08-2020
Author:https://github.com/bernard0047
'''
import os
import shutil
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import argparse
import re
#RAMESH KUMAR
print("HELLO GUYS, LET US MAKE AN INTERSTING PROGRAM")

def get_parser():
    parser = argparse.ArgumentParser(description='parameters for dataset preprocessing')
    parser.add_argument('--input_dir', default="./IMAGES3/", help='Input path (contains imgfolder and label csv)')
    parser.add_argument('--output_dir', default="./images/", help='a folder containing train and test folders will be created here') #don't pass for easy training
    args = parser.parse_args()
    return args

def label_check(label):
    if len(label)< 8:
        return 0

    if label[0:2]=='DL':
        delhi_pt = "\d{4,4}$"
        dlval = re.search(delhi_pt,label)
        if dlval is None or len(dlval.group())<8:
            return 0
        else:
            return 1
    pattern = "(([A-Za-z]){2,3}(|-)(?:[0-9]){1,2}(|-)(?:[A-Za-z]){2}(|-)([0-9]){1,4})|(([A-Za-z]){2,3}(|-)([0-9]){1,4})"
    val = re.search(pattern,label)
    if val is None or len(val.group())<8:
        return 0
    else:
        return 1

def size_check(ipath):
    img = cv.imread(ipath)
    height = img.shape[0]
    width = img.shape[1]
    #print(height,width)
    if height<24 or width<90:
        return 0
    return 1


def preprocess():
    args = get_parser()
    idr = os.path.expanduser(args.input_dir)
    odr = os.path.expanduser(args.output_dir)
    if os.path.exists(odr):
        print("Error: Path exists")
        return
    else:
        os.mkdir(odr)
    #print(input_dir,output_dir)
    imgfolder,dfpath = None, None
    for item in os.listdir(idr):
        if os.path.isdir(os.path.join(idr,item)):
            imgfolder = os.path.join(idr,item)
        else:
            dfpath = os.path.join(idr,item)
    if dfpath[-3:]!="csv":
        df = pd.read_excel(dfpath)
    else:
        df = pd.read_csv(dfpath,encoding = "ISO-8859-1")
    df = df.astype(str)
    allFileNames = os.listdir(imgfolder)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                            [int(len(allFileNames)*0.9)])
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Testing: ', len(test_FileNames))
    train_FileNames = [imgfolder+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [imgfolder+'/' + name for name in test_FileNames.tolist()]
    os.makedirs(odr +'/train')
    os.makedirs(odr +'/test')
    for name in train_FileNames:
        shutil.copy(name, odr+"/train")
    for name in test_FileNames:
        shutil.copy(name, odr+"/test")
    count=0
    for dirs in os.listdir(odr):
        for img in os.listdir(os.path.join(odr+dirs)):
            ipath = os.path.join(odr,dirs,img)
            img,_ = os.path.splitext(img) #uncomment this line if your dataset has imgname without label and modify to accomodate imgname of type(int)
            _,ext = os.path.splitext(ipath)
            label = df[df.iloc[:,0]==img].iloc[0,1]
            label = ''.join(e for e in label if e.isalnum())
            if img not in df.iloc[:,0].tolist() or label_check(label)==0 or size_check(ipath)==0:
                count+=1
                print(f"Image not found/ Image too small Label error: Discarding image:{img}")
                os.remove(ipath)
                continue
            tpath = os.path.join(odr,dirs,label+ext)
            if os.path.exists(tpath):
                if os.path.getsize(tpath)>os.path.getsize(ipath):
                    os.remove(ipath)
                else :
                    os.remove(tpath)
                    os.rename(ipath,tpath)
                count+=1
                print(f"Discarding duplicate image:{img}")
                continue

            os.rename(ipath,tpath)
    print(f"Discarded {count} images in total.")
            

if __name__ == "__main__":
    preprocess()






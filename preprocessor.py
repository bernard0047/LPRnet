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
#adding animated titles
# importing the necessary packages 
import time 
import sys 
import os 
  
# Function for implementing the loading animation 
def load_animation(): 
  
    # String to be displayed when the application is loading 
    load_str = "starting your console application..."
    ls_len = len(load_str) 
  
  
    # String for creating the rotating line 
    animation = "|/-\\"
    anicount = 0
      
    # used to keep the track of 
    # the duration of animation 
    counttime = 0        
      
    # pointer for travelling the loading string 
    i = 0                     
  
    while (counttime != 100): 
          
        # used to change the animation speed 
        # smaller the value, faster will be the animation 
        time.sleep(0.075)  
                              
        # converting the string to list 
        # as string is immutable 
        load_str_list = list(load_str)  
          
        # x->obtaining the ASCII code 
        x = ord(load_str_list[i]) 
          
        # y->for storing altered ASCII code 
        y = 0                             
  
        # if the character is "." or " ", keep it unaltered 
        # switch uppercase to lowercase and vice-versa  
        if x != 32 and x != 46:              
            if x>90: 
                y = x-32
            else: 
                y = x + 32
            load_str_list[i]= chr(y) 
          
        # for storing the resultant string 
        res =''              
        for j in range(ls_len): 
            res = res + load_str_list[j] 
              
        # displaying the resultant string 
        sys.stdout.write("\r"+res + animation[anicount]) 
        sys.stdout.flush() 
  
        # Assigning loading string 
        # to the resultant string 
        load_str = res 
  
          
        anicount = (anicount + 1)% 4
        i =(i + 1)% ls_len 
        counttime = counttime + 1
      
    # for windows OS 
    if os.name =="nt": 
        os.system("cls") 
          
    # for linux / Mac OS 
    else: 
        os.system("clear") 
  
# Driver program 
if __name__ == '__main__':  
    load_animation() 
  
    # Your desired code continues from here  
    # s = input("Enter your name: ") 
    s ="David"
    sys.stdout.write("Hello "+str(s)+"\n") 

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






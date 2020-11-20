# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from sklearn.metrics import classification_report
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import time
import cv2
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--test_img_dirs', default="./1line/train", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=15, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=64, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--savepreds', default=False, type=bool, help='save incorrect testpreds with original label to ./testpreds')
    parser.add_argument('--evaluate', default=True, type=bool, help='Find confusion matrix and classification report and save results')
    parser.add_argument('--pretrained_model', default='./weights/toppermodel0.15.pth', help='pretrained base model')
    parser.add_argument('--postprocess', default=True, type=bool, help='Apply postprocessing steps')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    filenames = []
    for _, sample in enumerate(batch):
        img, label, length, filename = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
        filenames.append(filename)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths, filenames)

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
    c_matrix = np.zeros((len(CHARS)-1,len(CHARS)-1))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t_chars = 0
    T_c = 0
    T_f = 0
    T_fc = 0
    t_fchars = 0
    res_chars = np.zeros(len(CHARS))
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths, filenames = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        #print(len(preb_labels))
    

        for i, label in enumerate(preb_labels):
            if args.postprocess:
                label = postprocess(label)
            correct=False
            X=i
            lb = ""
            tg = ""
            for j in targets[i]:
                x = int(j)
                tg+= CHARS[x]
            for j in label:
                lb += CHARS[j]

            # show image and its predict label
            t_chars+=len(targets[i])
            for j in range(len(label)):
                if j>=len(targets[i]):
                    continue
                if label[j] == targets[i][j]:
                    res_chars[label[j]]+=1
                    T_c+=1
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                #print(abs(len(label)-len(targets[i])))
                Tn_1 += 1

            else:
                c_matrix = cmatrix(c_matrix,label,targets[i])
                t_fchars+=len(targets[i])
                for j in range(len(label)):
                    if j>=len(targets[i]):
                        continue
                    if label[j] == targets[i][j]:
                        res_chars[label[j]]+=1
                        T_fc+=1
                fuzzy = 0
                for x in range(len(label)):
                    if targets[i][x]==label[x]:
                        fuzzy += 1
                if fuzzy/len(label) >= 0.75:
                    T_f += 1
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                    correct=True
                else:
                    Tn_2 += 1
            # print(lb,tg)
            if args.savepreds:
                if not os.path.isdir('./testpreds'):
                    os.makedirs('./testpreds')
                    os.makedirs('./testpreds/images')
                newname = 'testpreds/images/'+filenames[X].split('\\')[1].split('.')[0]+"__"+lb+'.png'
                if not correct:
                    shutil.copy(filenames[X],newname)
    if args.evaluate:
        evaluate_and_save(c_matrix)

            
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    print(f"[Info] 75%+ Accuracy: {T_f/(Tp+Tn_1+Tn_2)} [{T_f}/{(Tp+Tn_1+Tn_2)}]")
    t2 = time.time()
    print(f'[Info] Global Char Accuracy:{T_c/t_chars} [{T_c}/{t_chars}] ')
    print(f'[Info] Char Accuracy on full length match:{T_fc/t_fchars} [{T_fc}/{t_fchars}] ')
    print(f"Length accuracy: {(Tp+Tn_2)/(Tp+Tn_1+Tn_2)}")
    # print('Per char: ')
    # for i in range(10):
    #     print(i,": ",res_chars[i]/T_c)
    # for i in range(10,len(CHARS)-1):
    #     print(chr(55+i),': ',res_chars[i]/T_c)
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


def cmatrix(matrix,label,target):
    for i in range(len(label)):
        pred = int(label[i])
        tg = int(target[i])
        matrix[pred][tg]+=1
    return matrix

def evaluate_and_save(matrix):
    if not os.path.isdir('./testpreds'):
        os.makedirs('./testpreds')
    df = pd.DataFrame(matrix)
    df['index'] = CHARS[:-1]
    df.set_index('index',drop=True,inplace=True)
    df.to_csv("./testpreds/confusion_matrix.csv",header=CHARS[:-1])
    y_true = []
    y_pred = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            for _ in range(int(matrix[i][j])):
                y_true.append(j)
                y_pred.append(i)
    report = classification_report(y_true, y_pred, target_names=CHARS[:-1], output_dict=True)
    pd.DataFrame(report).transpose().to_csv('./testpreds/Cls_reports.csv')

def postprocess(label):
    if len(label)!=10:
        return label
    lb = ''
    for j in label:
        x = int(j)
        lb+= CHARS[x]
    
    newlb = correct(lb)
    label = []
    for ch in newlb:
        label.append(CHARS.index(ch))
    return label
    

def show(img, label, target):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    cv2.imshow("test", img)
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def correct(lb):
    return lb
    if lb[:2]=='DL':
        return lb
    abc = lb[:2]+lb[4:6]
    num = lb[2:4]+lb[6:10]
    num.replace('D','0')
    alphabet = ['A','B','D','I','O']

    if 'A' in num:
        num = num.replace('A','4')
    if 'B' in num:
        num = num.replace('B','8')
    if 'O' in num or 'D' in num:
        num = num.replace('O','0')
        num = num.replace('D','0')
    if 'I' in num:
        num = num.replace('I','1')
    
    if '4' in abc:
        abc = abc.replace('4','A')
    newlb = abc[:2]+num[:2]+abc[2:4]+num[2:]
    return newlb     

if __name__ == "__main__":
    test()

#Color dict
import numpy as np
# import wandb
import cv2 
import torch
import os

colors = {
    '0':[(128, 64, 128), (244, 35, 232), (0, 0, 230), (220, 190, 40), (70, 70, 70), (70, 130, 180), (0, 0, 0)],
    '1':[(128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140), (220, 20, 60), (255, 0, 0), (0, 0, 230), (255, 204, 54), (0, 0, 70), (220, 190, 40), (190, 153, 153), (174, 64, 67), (153, 153, 153), (70, 70, 70), (107, 142, 35), (70, 130, 180)], 
    '2':[(128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140), (220, 20, 60), (255, 0, 0), (0, 0, 230), (119, 11, 32), (255, 204, 54), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (220, 190, 40), (102, 102, 156), (190, 153, 153), (180, 165, 180), (174, 64, 67), (220, 220, 0), (250, 170, 30), (153, 153, 153), (169, 187, 214), (70, 70, 70), (150, 100, 100), (107, 142, 35), (70, 130, 180)], 
    '3':[(128, 64, 128), (250, 170, 160), (81, 0, 81), (244, 35, 232), (230, 150, 140), (152, 251, 152), (220, 20, 60), (246, 198, 145), (255, 0, 0), (0, 0, 230), (119, 11, 32), (255, 204, 54), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (136, 143, 153), (220, 190, 40), (102, 102, 156), (190, 153, 153), (180, 165, 180), (174, 64, 67), (220, 220, 0), (250, 170, 30), (153, 153, 153), (153, 153, 153), (169, 187, 214), (70, 70, 70), (150, 100, 100), (150, 120, 90), (107, 142, 35), (70, 130, 180), (169, 187, 214), (0, 0, 142)]
    }

def visualize(mask,n_classes,ignore_label,gt = None):
    if(n_classes<len(colors['0'])):
        id = 0
    elif(n_classes<len(colors['1'])):
        id = 1
    elif(n_classes<len(colors['2'])):
        id = 2
    else:
        id = 3
    out_mask = np.zeros((mask.shape[0],mask.shape[1],3))
    for i in range(n_classes):
        out_mask[mask == i] = colors[str(id)][i]
    if(gt is not None):
        out_mask[gt == ignore_label] = (255,255,255)
    out_mask[np.where((out_mask == [0, 0, 0]).all(axis=2))] = (255,255,255)
    return out_mask

def error_map(pred,gt,cfg):
    canvas = pred.copy()
    canvas[canvas == gt] = 255
    canvas[gt == cfg.Loss.ignore_label] = 255
    return canvas

# def segmentation_validation_visualization(epoch,sample,pred,batch_size,class_labels,wandb_image,cfg):
    
#     os.makedirs(os.path.join(cfg.train.output_dir,'Visualization',str(epoch)),exist_ok = True)
    
#     input = sample['image'].permute(0,2,3,1).detach().cpu().numpy()
#     label = sample['label'].detach().cpu().numpy().astype(np.uint8)
#     pred = torch.argmax(pred[0],dim = 1).detach().cpu().numpy().astype(np.uint8)
    
#     for i in range(batch_size):
#         errormap = error_map(pred[i],label[i],cfg)
#         wandb_image.append(wandb.Image(cv2.resize(cv2.cvtColor(input[i], cv2.COLOR_BGR2RGB),(cfg.dataset.width//4,cfg.dataset.height//4)), masks={
#                                             "predictions" : {
#                                                 "mask_data" : cv2.resize(pred[i],(cfg.dataset.width//4,cfg.dataset.height//4)),
#                                                 "class_labels" : class_labels
#                                             },
#                                             "ground_truth" : {
#                                                 "mask_data" :  cv2.resize(label[i],(cfg.dataset.width//4,cfg.dataset.height//4)),
#                                                 "class_labels" : class_labels
#                                              }
#                                             ,
#                                              "error_map" : {
#                                                  "mask_data" :  cv2.resize(errormap,(cfg.dataset.width//4,cfg.dataset.height//4)),
#                                                  "class_labels" : class_labels
#                                              }
#                                         }))
        
#         if(cfg.valid.write):
#             prediction = visualize(pred[i],cfg.model.n_classes,cfg.Loss.ignore_label,gt = label[i])  
#             mask = visualize(label[i],cfg.model.n_classes,cfg.Loss.ignore_label,gt = label[i])
#             out = np.concatenate([((input[i]* np.array(cfg.dataset.mean) + np.array(cfg.dataset.std))*255).astype(int),mask,prediction,visualize(errormap,cfg.model.n_classes,cfg.Loss.ignore_label,label[i])],axis = 1)
#             cv2.imwrite(os.path.join(cfg.train.output_dir,'Visualization',str(epoch),sample['img_name'][i]),out)
#     return wandb_image

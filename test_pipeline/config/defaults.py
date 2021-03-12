from yacs.config import CfgNode as CN
import numpy as np

_C = CN()
# dataset
_C.dataset = CN()
_C.dataset.height = 175
_C.dataset.width = 175
_C.dataset.mean = [0.485, 0.456, 0.406]
_C.dataset.std = [0.229, 0.224, 0.225]
_C.dataset.batch_size_pergpu = 4
_C.dataset.shuffle = True
_C.dataset.csvpath = ""
_C.dataset.num_workers = 8
_C.dataset.data_name = ""
_C.dataset.segm_downsampling_rate = 4

_C.dataset.augmentation = CN()
_C.dataset.augmentation.augmix = CN()
_C.dataset.augmentation.cutout = CN()
_C.dataset.augmentation.techniques = CN()
_C.dataset.augmentation.techniques.pixel = CN()
_C.dataset.augmentation.techniques.spatial = CN()

_C.dataset.augmentation.augmix.val = False
_C.dataset.augmentation.augmix.width = 3
_C.dataset.augmentation.augmix.depth = 4
_C.dataset.augmentation.augmix.alpha = 1

_C.dataset.augmentation.cutout.val = False
_C.dataset.augmentation.cutout.n_holes = 20
_C.dataset.augmentation.cutout.length = 150

_C.dataset.augmentation.techniques.pixel = CN()

_C.dataset.augmentation.techniques.pixel.RandomBrightnessContrast = False
_C.dataset.augmentation.techniques.pixel.Blur = False
_C.dataset.augmentation.techniques.pixel.OpticalDistortion = False
_C.dataset.augmentation.techniques.pixel.ImageCompression = False
_C.dataset.augmentation.techniques.pixel.MultiplicativeNoise = False
_C.dataset.augmentation.techniques.pixel.IAASharpen = False
_C.dataset.augmentation.techniques.pixel.MotionBlur = False
_C.dataset.augmentation.techniques.pixel.IAAEmboss = False
_C.dataset.augmentation.techniques.pixel.MedianBlur = False
_C.dataset.augmentation.techniques.pixel.GaussNoise = False
_C.dataset.augmentation.techniques.pixel.RandomGamma = False
_C.dataset.augmentation.techniques.pixel.HueSaturationValue = False
_C.dataset.augmentation.techniques.pixel.CLAHE = False

_C.dataset.augmentation.techniques.spatial = CN()

_C.dataset.augmentation.techniques.spatial.verticalflip = False
_C.dataset.augmentation.techniques.spatial.horizontalflip = False
_C.dataset.augmentation.techniques.spatial.randomcrop = False
_C.dataset.augmentation.techniques.spatial.scale = False
_C.dataset.augmentation.techniques.spatial.scale_factor = 1

_C.dataset.augmentation.techniques.spatial.cropping = CN()
_C.dataset.augmentation.techniques.spatial.cropping.width = 150
_C.dataset.augmentation.techniques.spatial.cropping.height = 150


# train
_C.train = CN()

_C.train.n_epochs = 30
_C.train.gpus = (0,)
_C.train.accumulation_steps = 1
_C.train.output_dir = "./ckpts"
_C.train.config_path = "config/config.yaml"
_C.train.n_iterations = 30
# validate
_C.valid = CN()
_C.valid.val = False
_C.valid.frequency = 1
_C.valid.n_samples_visualize = 50
_C.valid.write = (
    False  # To write predictions in local system files uploaded to wandb by default
)

# model
_C.model = CN()
_C.model.OCR = CN()

_C.model.backbone = "hrnetv2"  # add extra in config file you are passing if you want to use hrnet, samples given in config for hrnet18 and hrnet48
_C.model.fcdim = 720  # Final layer dimension of current backbone
_C.model.n_classes = 4
_C.model.amp = False
_C.model.decoder = "ocr"
_C.model.activation = "mish"
_C.model.OCR.MID_CHANNELS = 256
_C.model.OCR.KEY_CHANNELS = 512
_C.model.EXTRA = CN(new_allowed=True)
_C.model.pretrained = "/home/sanchit/Workspace/semantic-segmentation-pipeline-master/hrnet_w18_small_v2_cityscapes_cls19_1024x2048_trainset.pth"
# optimizer
_C.optimizer = CN()
_C.optimizer.lrscheduler = CN()

_C.optimizer.val = "ranger"
_C.optimizer.lr = 0.0005
_C.optimizer.weight_decay = 0.0001
_C.optimizer.gradientcentralization = False
_C.optimizer.lrscheduler.val = "multisteplr"
_C.optimizer.lrscheduler.param = [10, 20]
# loss
_C.Loss = CN()
_C.Loss.val = "ocrloss"
_C.Loss.class_weights = [1]
_C.Loss.ignore_label = 255
_C.Loss.gamma = 1.0
_C.Loss.alpha = 1.0
_C.Loss.rate = 0.7


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

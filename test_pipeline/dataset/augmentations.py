import cv2
from albumentations import (
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    RandomBrightnessContrast,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    Flip,
    OneOf,
    Compose,
    Resize,
    ImageCompression,
    MultiplicativeNoise,
    ChannelDropout,
    IAASuperpixels,
    GaussianBlur,
    HorizontalFlip,
    RandomGamma,
    VerticalFlip,
    ShiftScaleRotate,
    CLAHE,
    RandomResizedCrop,
    CenterCrop,
    ShiftScaleRotate,
)

import numpy as np
import torch
from torchvision import transforms
import random

augmentation_pixel_techniques_pool = {
    "RandomBrightnessContrast": RandomBrightnessContrast(
        brightness_limit=(0.2, 0.6), contrast_limit=0.4, p=1
    ),
    "Blur": Blur(blur_limit=2, p=1),
    "OpticalDistortion": OpticalDistortion(distort_limit=0.20, shift_limit=0.15, p=1),
    "ImageCompression": ImageCompression(p=1),
    "MultiplicativeNoise": MultiplicativeNoise(multiplier=(0.5, 5), p=1),
    "IAASharpen": IAASharpen(alpha=(0.2, 1), lightness=(0.5, 1.0), p=1),
    "IAAEmboss": IAAEmboss(alpha=(0.2, 1), strength=(0.5, 1.0), p=1),
    "MotionBlur": MotionBlur(blur_limit=15, p=1),
    "MedianBlur": MedianBlur(blur_limit=7, p=1),
    "GaussNoise": GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1),
    "RandomGamma": RandomGamma(gamma_limit=(30, 120), p=1),
    "HueSaturationValue": HueSaturationValue(
        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1
    ),
    "CLAHE": CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
}


class Spatial_augmentation:
    """
    #todo
    add rotation
    add shift
    """

    def apply(self, cfg, image, label=None):

        self.base_size = max(cfg.dataset.height, cfg.dataset.width)
        self.ignore_label = cfg.Loss.ignore_label

        # apply scaling

        if cfg.dataset.augmentation.techniques.spatial.scale:
            image, label = self.multi_scale_aug(
                image,
                label,
                scale_factor=cfg.dataset.augmentation.techniques.spatial.scale_factor,
            )

        # apply random cropping

        if cfg.dataset.augmentation.techniques.spatial.randomcrop:
            image, label = self.rand_crop(
                image,
                label,
                crop_size=(
                    cfg.dataset.augmentation.techniques.spatial.cropping.height,
                    cfg.dataset.augmentation.techniques.spatial.cropping.width,
                ),
            )

        # apply horizontal flip
        if cfg.dataset.augmentation.techniques.spatial.horizontalflip:
            image, label = self.horizontalflip(image, label)
        # apply vertical flip

        if cfg.dataset.augmentation.techniques.spatial.verticalflip:
            image, label = self.verticalflip(image, label)

        if label is not None:
            label = label.astype("int32")
            if cfg.dataset.segm_downsampling_rate > 1:
                label = cv2.resize(
                    label,
                    None,
                    fx=1 / cfg.dataset.segm_downsampling_rate,
                    fy=1 / cfg.dataset.segm_downsampling_rate,
                    interpolation=cv2.INTER_NEAREST,
                )
            return image, label

        return image

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
            )
        return pad_image

    def rand_crop(self, image, label, crop_size):

        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, crop_size, (0.0, 0.0, 0.0))

        new_h, new_w = image.shape[:-1]
        x = random.randint(0, new_w - crop_size[1])
        y = random.randint(0, new_h - crop_size[0])

        image = image[y : y + crop_size[0], x : x + crop_size[1]]

        if label is not None:
            label = self.pad_image(label, h, w, crop_size, (self.ignore_label,))
            label = label[y : y + crop_size[0], x : x + crop_size[1]]
        return image, label

    def multi_scale_aug(self, image, label, scale_factor=1):

        rand_scale = 0.5 + random.randint(0, scale_factor) / 10.0
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return image, label

    def horizontalflip(self, image, label):
        flip = np.random.choice(2) * 2 - 1
        if flip:
            image = image[:, ::flip, :]
            if label is not None:
                label = label[:, ::flip]
        return image, label

    def verticalflip(self, image, label):
        flip = np.random.choice(2) * 2 - 1
        image = image[::flip, :, :]
        if label is not None:
            label = label[::flip, :]

        return image, label


def apply_op(image, op, mask=None):
    """
    Apply augmentation function specifically for augmix

    Arguments:
        image {[Numpy array]} -- [description]
        op {[list]} -- [list of all the augmentations to be applied sequentially]

    Returns:
        [type] -- [transformed image]
    """
    return op(image=image, mask=mask)


def augment_and_mix(image, augs, cfg):

    """
    Augmix - https://arxiv.org/abs/1912.02781

    Arguments:
        image {[numpy array]} -- []
        augs {[list of function(augmentations)]} -- [List of all augmentations applied to dataset]
        cfg {[Config File]} -- []
            Augmix hyperparameters:
            width {[int]} -- [Number of parallel augmentation paths]
            depth {[int]} -- [Number of augmentations applied to each image in each path]
            alpha {[float (0-1)]} -- [Probability coefficient for Beta and Dirichlet distributions.]
        tranform {[torchvision.compose]} -- [Pytorch transform for normalization]
    Returns:
        [torch tensor] -- [Transformed and normalized image]
    """

    width = cfg.dataset.augmentation.augmix.width
    depth = cfg.dataset.augmentation.augmix.depth
    alpha = cfg.dataset.augmentation.augmix.alpha
    ops = []
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    for i in range(width):
        op = []
        ag = augs.copy()
        for j in range(depth):
            a = np.random.choice(ag)
            ag.remove(a)
            op.append(a)
        ops.append(Compose(op))

    if cfg.dataset.augmentation.techniques.spatial.randomcrop:
        mix = np.zeros(
            (
                cfg.dataset.augmentation.techniques.spatial.cropping.height,
                cfg.dataset.augmentation.techniques.spatial.cropping.width,
                3,
            )
        )
    else:
        mix = np.zeros((cfg.dataset.height, cfg.dataset.width, 3))

    for i in range(width):
        image_aug = image.copy()
        op = ops[i]
        augmented_output = apply_op(image_aug, op)
        image_aug = augmented_output["image"]
        mix += ws[i] * image_aug
    output_image = (1 - m) * image + m * mix
    return output_image.astype(np.uint8)


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def normalize(image, cfg):
    return transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)(
        transformation(image)
    )

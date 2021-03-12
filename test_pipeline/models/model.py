from . import hrnet
from . import hrnet_cbam
from . import decoder as dec
from torch import nn as nn
import torch
import os
import torch.nn.functional as F
from .resnest import resnest101,resnest18
from . import hrnet_cbn
from . import hrnet_lambda

activation = {"mish": nn.ELU(), "relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU()}

# Similarly implement norm
# output of encoder and decoder should be list, can be of single element
encoders = {
    "hrnetv2": hrnet.hrnetv2,
    "resnest": resnest18,
    "hrnetv2cbn": hrnet_cbn.hrnetv2,
    "hrnetv2lambda": hrnet_lambda.hrnetv2,
    "hrnetv2cbam": hrnet_cbam.hrnetv2,
    
}


def create_model(cfg):

    encoder = encoders[cfg.model.backbone](cfg)
    decoder = dec.decoders[cfg.model.decoder](cfg=cfg)

    class model(nn.Module):
        def __init__(self, encoder, decoder):
            super(model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x, segSize=None):
            enc_out = self.encoder(x)
            out = self.decoder(enc_out, segSize)
            return {"enc_out": enc_out, "output": out}

    m = init_weights(model(encoder, decoder))
    if os.path.exists(cfg.model.pretrained):
        print("loading pretrained weight")
        b = load(m.state_dict(), torch.load(cfg.model.pretrained))
        m.load_state_dict(b)
    return m


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


def load(a, b):
    total_weights = 0
    for k, v in a.items():
        count = 0
        total_weights += 1
        flag = 0
        for k1, v1 in b.items():
            if ".".join(k.split(".")[1:]) == k1:
                try:
                    a[k].copy_(v1)
                    flag = 1
                except:
                    flag = 0
        if flag == 0:
            count += 1
            print("unmatched weight key:", k)
    print("number of unmatched weights:", count)
    return a

from torch import nn as nn
import torch
import torch.nn.functional as F
from .cbn import CBatchNorm2d, ConvModule

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

# output of decoders should be list, with first element being the one which will be used at the time of inference


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    if BatchNorm2d == nn.BatchNorm2d:

        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return ConvModule(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1
        )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    if BatchNorm2d == nn.BatchNorm2d:

        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
            BatchNorm2d(out_planes, momentum=0.01),
            nn.ReLU(inplace=False),
        )
    else:
        return ConvModule(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0
        )

class C1_transposed(nn.Module):
    def __init__(self, cfg, use_softmax=False):
        super(C1_transposed, self).__init__()
        self.use_softmax = use_softmax
        fc_dim = cfg.model.fcdim
        self.cbr = nn.Sequential(
            nn.ConvTranspose2d(fc_dim, fc_dim//2, kernel_size=2, stride=2, padding=0),
            BatchNorm2d(fc_dim//2, momentum=0.01),
            nn.ReLU(inplace=False),
        )
        self.conv_last = nn.ConvTranspose2d(fc_dim//2, cfg.model.n_classes, 2, 2, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if segSize:
            x = F.upsample(x, size=segSize, mode="bilinear")
        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
        return [x]
class C1(nn.Module):
    def __init__(self, cfg, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax
        fc_dim = cfg.model.fcdim
        self.cbr = conv1x1_bn_relu(fc_dim, fc_dim, 1)
        self.conv_last = nn.Conv2d(fc_dim, cfg.model.n_classes, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if segSize:
            x = F.upsample(x, size=segSize, mode="bilinear")
        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
        return [x]
class C1_context(nn.Module):
    def __init__(self, cfg, use_softmax=False):
        super(C1_context, self).__init__()
        self.use_softmax = use_softmax
        fc_dim = cfg.model.fcdim
        self.context = ContextModule(fc_dim,fc_dim//2)
        self.cbr = conv1x1_bn_relu(fc_dim*2, fc_dim*2, 1)
        self.conv_last = nn.Conv2d(fc_dim*2, cfg.model.n_classes, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        context = self.context(conv5)
        conv5 = torch.cat([conv5, context],1)
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if segSize:
            x = F.upsample(x, size=segSize, mode="bilinear")
        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
        return [x]


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(
            nn.Conv2d(inter_plane, plane, kernel_size=1), BatchNorm2d(plane)
        )

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q, node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out
class DualGCN(nn.Module):
    """
    Feature GCN with coordinate GCN
    """

    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(
            planes // ratio, planes // ratio, kernel_size=1, bias=False
        )
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(
            planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False
        )
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(
                planes, planes, 3, groups=planes, stride=2, padding=1, bias=False
            ),
            BatchNorm2d(planes),
            nn.Conv2d(
                planes, planes, 3, groups=planes, stride=2, padding=1, bias=False
            ),
            BatchNorm2d(planes),
            nn.Conv2d(
                planes, planes, 3, groups=planes, stride=2, padding=1, bias=False
            ),
            BatchNorm2d(planes),
        )
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
            BatchNorm2d(planes),
        )

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(
            local, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x + y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out

class ContextModule(nn.Module):
    '''
    this is essentialy a bi-LSTM that process the feature vectors.
    It recieves a (b, c, h, w) tensor and outputs a tensor
    of the same size after the rnn pass.
    :param input_size - number of channels in the input.
    :param hidden_size - dimension of the LSTM hidden layers.
    '''

    def __init__(self, input_size, hidden_size):
        super(ContextModule, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        bs, h, w, f = x.size()
        x = x.view(bs, h * w, f)
        x, _ = self.lstm(x)
        x = x.contiguous().view(bs, h, w, 2 * self.hidden_size)
        x = x.permute(0, 3, 1, 2)
        return x
class DualGCNHead(nn.Module):
    def __init__(self, cfg):
        super(DualGCNHead, self).__init__()
        inplanes = cfg.model.EXTRA.inplanes
        inplanes_dsn = cfg.model.EXTRA.inplanes_dsn
        interplanes_dsn = cfg.model.EXTRA.interplanes_dsn

        interplanes = cfg.model.EXTRA.interplanes
        num_classes = cfg.model.n_classes

        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                inplanes + interplanes,
                interplanes,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(
                interplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )
        self.dsn = nn.Sequential(
            nn.Conv2d(
                inplanes_dsn, interplanes_dsn, kernel_size=3, stride=1, padding=1
            ),
            BatchNorm2d(interplanes_dsn),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                interplanes_dsn,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x, segSize=None):
        x2, x3 = x
        output = self.conva(x3)
        output = self.dualgcn(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x3, output], 1))
        if segSize:
            output = F.upsample(output, size=segSize, mode="bilinear")
            dsn = self.dsn(x2)
        else:
            dsn = self.dsn(x2)
        return [output, dsn]


decoders = {"C1": C1, "dgcnet": DualGCNHead,'C1_context':C1_context,'C1_transposed':C1_transposed}

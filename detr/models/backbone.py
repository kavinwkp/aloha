# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

# from util.misc import NestedTensor, is_main_process
from detr.util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from .resnet_film import resnet18_film

import IPython
e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    实现了冻结的批量归一化操作，其中权重和偏差是固定的，而均值和方差是从运行中累积的，用于规范化输入数据。
    这对于某些模型（如ResNet系列）的性能很重要，以避免产生NaN值
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:    # False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],  # 是否替换ResNet的部分stride为dilation卷积
            pretrained=is_main_process(),   # pretrained 只有主进程加载预训练权重
            norm_layer=FrozenBatchNorm2d)   # 指定使用FrozenBatchNorm2d作为BatchNorm的层 TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """用于将骨干网络的输出特征与位置编码相结合"""
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)   # {'0':(8,512,15,20)}
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():  # 遍历骨干网络的输出字典，其中name是特征图的名称，x是对应的特征图
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))  # [(1,512,15,20)]

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0   # True 根据lr_backbone的值，判断是否训练骨干网络。如果lr_backbone大于零，则表示需要对骨干网络进行微调
    return_interm_layers = args.masks   # False 根据masks的值，判断是否需要返回骨干网络的中间层特征。如果masks为True，则表示需要返回中间层特征，否则只返回最后一层特征
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


# class ResNetFilmBackbone(nn.Module):
#     def __init__(self, embedding_name: str, pretrained: bool = False,
#                  film_config=None):
#         super().__init__()
#         self._pretrained = pretrained   # False
#         weights = 'IMAGENET1K_V1' if pretrained else None   # None
#         # if embedding_name in ('resnet34_film', 'resnet34'):
#         #     backbone = resnet34_film(weights=weights, film_config=film_config, pretrained=pretrained)
#         #     embedding_dim = 512
#         # elif embedding_name in ('resnet18_film', 'resnet18'):
#         backbone = resnet18_film(weights=weights, film_config=film_config, pretrained=pretrained)
#         embedding_dim = 512
#         # else:
#         #     raise NotImplementedError
#
#         self.resnet_film_model = backbone
#         self._embedding_dim = embedding_dim
#         self.resnet_film_model.fc = nn.Identity()
#         self.resnet_film_model.avgpool = nn.Identity()
#
#         self.num_channels = self._embedding_dim
#
#         # FiLM config
#         self.film_config = film_config
#         if film_config is not None and film_config['use']:
#             film_models = []
#             for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):  # [2,2,2,2] 每层都是2个block
#                 if layer_idx in film_config['use_in_layers']:   # [1,2,3] 只在1,2,3层有用到，0层没用到
#                     num_planes = self.resnet_film_model.film_planes[layer_idx]  # [64,128,256,512]
#                     film_model_layer = nn.Linear(
#                         film_config['task_embedding_dim'], num_blocks * 2 * num_planes)
#                 else:
#                     film_model_layer = None
#                 film_models.append(film_model_layer)
#
#             self.film_models = nn.ModuleList(film_models)   # [None, Linear(512,512), Linear(512,1024), Linear(512,2048)]
#
#     def forward(self, x, task_emb, **kwargs):
#         film_outputs = None
#         if self.film_config is not None and self.film_config['use']:
#             film_outputs = []
#             for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):  # [2,2,2,2]
#                 if self.film_config['use'] and self.film_models[layer_idx] is not None:
#                     film_features = self.film_models[layer_idx](task_emb)   # 要先对tack_emb做个映射
#                 else:
#                     film_features = None
#                 film_outputs.append(film_features)  # [None, (bs,512), (bs,1024), (2,2048)]
#         return self.resnet_film_model(x, film_features=film_outputs, flatten=False)     # 再与resnet的中间结果运算
#
#     @property
#     def embed_dim(self):
#         return self._embedding_dim


# def build_film_backbone(args):
#     position_embedding = build_position_encoding(args)
#     film_config = {
#         'use': True,
#         'use_in_layers': [1, 2, 3],     # 只在resnet的1,2,3层使用，0层没用到
#         'task_embedding_dim': 512,
#         'film_planes': [64, 128, 256, 512],
#     }
#     backbone = ResNetFilmBackbone(args.backbone, film_config=film_config)
#     print(backbone)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model
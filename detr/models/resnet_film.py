from typing import Type, Any, Callable, Union, List, Mapping, Optional

import copy
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet18_Weights


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, film_features):
        identity = x    # layer2_block1:(bs,64,60,106), layer2_block2:(bs,128,30,53)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)     # layer2:(bs,128,30,53) layer3:(bs,256,15,27) layer4:(bs,512,8,14)
        # Apply FiLM here
        if film_features is not None:
            # gamma, beta will be (B, 1, 1, planes)
            gamma, beta = torch.split(film_features, 1, dim=1)  # layer2:(bs,1,1,128) layer3:(bs,1,1,256) layer3:(bs,1,1,512)
            gamma = gamma.squeeze().view(x.size(0), -1, 1, 1)   # layer2:(bs,128,1,1) layer3:(bs,256,1,1) layer4:(bs,512,1,1)
            beta = beta.squeeze().view(x.size(0), -1, 1, 1)     # layer2:(bs,128,1,1) layer3:(bs,256,1,1) layer4:(bs,512,1,1)
            out = (1 + gamma) * out + beta

        if self.downsample is not None:
            identity = self.downsample(x)   # (bs,128,30,53) layer2-4的第1个block才需要

        out += identity
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion: int = 4
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#
#         out = self.relu(out)
#
#         return out


class ResNetWithExtraModules(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            film_config=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.layers = layers

        self.use_film = film_config is not None and film_config['use']
        if self.use_film:
            self.film_config = film_config
            self.film_planes = film_config['film_planes']

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        in_channels_conv1 = 1 if (
            film_config is not None and
            film_config.get('append_object_mask', None) is not None) else 3     # 3

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels_conv1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  num_blocks=layers[0])
        self.layer2 = self._make_layer(block, 128, num_blocks=layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, num_blocks=layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, num_blocks=layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m_name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # if dilate:
        #     self.dilation *= stride
        #     stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = [
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        if self.use_film:
            return nn.ModuleList(layers)
        else:
            return nn.Sequential(*layers)

    def _forward_impl_film(self, x, film_features, flatten=True):
        """
        :param film_features: [None, (bs,512), (bs,1024), (bs,2048)]
        :param flatten: False
        """
        assert self.use_film and film_features is not None

        def _extract_film_features_for_layer(film_feat, layer_idx):
            if film_features[layer_idx] is None:
                return [None] * self.layers[layer_idx]

            num_planes = self.film_planes[layer_idx]    # [64, 128, 256, 512]
            num_blocks = self.layers[layer_idx]     # 2
            film_feat = film_feat.view(-1, 2, num_blocks, num_planes)   # layer2: (bs,2,2,128) layer3: (bs,2,2,256) layer4: (bs,2,2,512)
            film_feat_per_block = torch.split(film_feat, 1, dim=2)  # layer2: {(bs,2,1,128),..} layer3: {(bs,2,1,256),..} layer4: {(bs,2,1,512),..}
            return film_feat_per_block

        for layer_idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            film_feat_per_block = _extract_film_features_for_layer(film_features[layer_idx], layer_idx)
            # layer1:[None,None] layer2: {(bs,2,1,128),..} layer3: {(bs,2,1,256),..} layer4: {(bs,2,1,512),..}

            for block_idx, block in enumerate(layer):
                # if film_feat_per_block[block_idx] is not None:
                #     assert x.shape[0] == film_feat_per_block[block_idx].shape[0], ('FiLM batch size does not match')
                x = block(x, film_features=film_feat_per_block[block_idx])

        x = self.avgpool(x)     # (2,512,8,14)
        if flatten:     # False
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _forward_impl(self, x, film_features, flatten):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.use_film:
            return self._forward_impl_film(x, film_features, flatten)

    def forward(self, x, film_features, **kwargs):
        return self._forward_impl(x, film_features, **kwargs)


def _resnet(block, layers, weights, progress, **kwargs):
    model_kwargs = copy.deepcopy(kwargs)
    if 'pretrained' in model_kwargs:
        del model_kwargs['pretrained']
    if 'arch' in model_kwargs:
        del model_kwargs['arch']
    model = ResNetWithExtraModules(block, layers, **model_kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def resnet18_film(*, weights, progress=True, **kwargs):
    weights = ResNet18_Weights.verify(weights)
    return _resnet(block=BasicBlock, layers=[2, 2, 2, 2], weights=weights, progress=progress, **kwargs)
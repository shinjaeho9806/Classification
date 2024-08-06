import torch
from torch import nn, Tensor
import math

def conv_3x3_bn(in_channels, out_channels, stride):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )
    return block

def conv_1x1_bn(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )
    return block

class InvertedResidual(nn.Module):
    '''
    hyper_parameter
    * expand ratio -> t
    '''
    def __init__(self, in_channels, out_channels, stride, t):
        super().__init__()
        assert stride in [1,2]
        
        hidden_channels = round(in_channels * t)
        # True / False skip connection 사용 유무, 채널수 변화로 확인
        self.identity = stride == 1 and in_channels == out_channels
        
        if t == 1:
            self.block = nn.Sequential(
                # depthwise
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size = 3, stride = stride, padding = 1, groups = in_channels, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # pointwise-linear
                nn.Conv2d(hidden_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                # pointwise
                nn.Conv2d(in_channels, hidden_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # depthwise
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size = 3, stride = stride, padding = 1, groups = hidden_channels, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # pointwise-linear
                nn.Conv2d(hidden_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        if self.identity:
            return self.block(x) + x
        else:
            return self.block(x)

class MobileNetV2(nn.Module):
    '''
    hyper_parameter 
    * expand ratio -> t
    * width multiplier -> alpha
    * resolution multiplier -> roh
    '''
    def __init__(self, num_classes, t = 1, alpha = 1.):
        super().__init__()
        self.cfgs = [
            # t, c, n, s -> expansion_factor, out_channels, 반복수, stride
            [1,16,1,1],
            [6,24,2,2],
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,1,1]
        ]
        
        in_channels = _make_divisible(32 * alpha, 4 if alpha == 0.1 else 8)
        self.conv1 = conv_3x3_bn(3, in_channels, 2)
        layers = []
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            out_channels = _make_divisible(c * alpha, 4 if alpha == 0.1 else 8)
            for i in range(n):
                layers.append(block(in_channels, out_channels, s if i == 0 else 1, t))
                in_channels = out_channels
        self.bottleneck = nn.Sequential(*layers)
        out_channels = _make_divisible(1280 * alpha, 4 if alpha == 0.1 else 8)
        self.conv2 = conv_1x1_bn(in_channels, out_channels)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(out_channels, num_classes)
        )
        self._init_layers()
            
    def _make_divisible(v, divisor, min_value = None):
        '''
        모든 채널이 8로 나누어 질수 있도록!!
        '''
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_value < .9 * new_value:
            new_value += divisor
        return new_value
        
    def _init_layers(self):
        pass
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bottleneck(out)
        out = self.conv2(out)
        out = self.avg(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out
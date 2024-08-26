import torch
from torch import nn, Tensor

def _make_divisible(v, divisor, min_value = None):
    if min_value is None:
        min_value = divisor
    
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_relu6(nn.Module):
    '''
    h_relu = relu6(x + 3) / 6
    '''
    def __init__(self, ):
        super().__init__()
        self.relu = nn.ReLU6(inplace = inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    '''
    * mobilenetv3에서 제안한 h_swish
    * h_swish = x * h_relu6
    '''
    def __init__(self, ):
        super().__init__()
        self.sigmoid = h_relu6(inplace = inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, in_channel, reduction = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channel, _make_divisible(in_channel // reduction, 8)),
            nn.ReLU(),
            nn.Linear(_make_divisible(in_channel // reduction, 8) ,in_channel),
            h_relu6()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.squeeze(x).view(b,c)
        out = self.excitation(out).view(b,c,1,1)
        return x * out
    
def conv_3x3_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        h_swish()
    )

def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        h_swish()
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1,2]
        
        self.identity = stride == 1 and in_channels == out_channels
        
        if in_channels == inner_channels:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, kernel_size = kernel_size, stride = stride, padding = (kernel_size - 1) // 2, groups = inner_channels, bias = False),
                nn.BatchNorm2d(inner_channels),
                h_swish() if use_hs else nn.ReLU(),
                SELayer(inner_channels) if use_se else nn.Identity(),
                nn.Conv2d(inner_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(inner_channels),
                h_swish() if use_hs else nn.ReLU(),
                nn.Conv2d(inner_channels, inner_channels, kernel_size = kernel_size, stride = stride, padding = (kernel_size - 1) // 2, groups = inner_channels, bias = False),
                nn.BatchNorm2d(inner_channels),
                SELayer(inner_channels) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(),
                nn.Conv2d(inner_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    '''
    width_mult = MobileNetV1의 alpha와 같은 역할
    '''
    def __init__(self, cfgs, mode, num_classes = 1000, width_mult = 1., block = InvertedResidual):
        super().__init__()
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        
        inner_channels = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, inner_channels, stride = 2)]
        for k, t, c, use_se, use_hs, s in self.cfgs:
            '''
            k = kernel_size
            t = 확장계수
            c = out_channels
            use_se, use_hs = SELayer, H_Swish사용유무
            s = stride
            '''
            out_channels = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(inner_channels * t, 8)
            layers.append(
                block(in_channels = inner_channels, 
                      inner_channels = exp_size,
                      out_channels = out_channels, 
                      kernel_size = k, stride = s, 
                      use_se = use_se, use_hs = use_hs)
            )
            inner_channels = out_channels
        self.block = nn.Sequential(*layers)
        self.last_conv = conv_1x1_bn(inner_channels, exp_size)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        out_channels = {'large' : 1280, 'small' : 1024}
        out_channels = _make_divisible(out_channels[mode] * width_mult, 8) if width_mult > 1. else out_channels[mode]
        self.fc = nn.Sequential(
            nn.Linear(exp_size, out_channels),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, num_classes)
        )
        self._init_layers()
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        out = self.block(x)
        out = self.last_conv(out)
        out = self.avg(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def mobilenetv3_large(**kwargs):
    cfgs = [
        # kernel_size, t, out_channels, SE, HS, stride
        [3,   1,  16, False, False, 1],
        [3,   4,  24, False, False, 2],
        [3,   3,  24, False, False, 1],
        [5,   3,  40, True, False, 2],
        [5,   3,  40, True, False, 1],
        [5,   3,  40, True, False, 1],
        [3,   6,  80, False, True, 2],
        [3, 2.5,  80, False, True, 1],
        [3, 2.3,  80, False, True, 1],
        [3, 2.3,  80, False, True, 1],
        [3,   6, 112, True, True, 1],
        [3,   6, 112, True, True, 1],
        [5,   6, 160, True, True, 2],
        [5,   6, 160, True, True, 1],
        [5,   6, 160, True, True, 1]
    ]
    return MobileNetV3(cfgs, mode = 'large', **kwargs)

def mobilenetv3_small(**kwargs):
    cfgs = [
        # kernel_size, t, out_channels, SE, HS, stride
        [3,    1,  16, True, False, 2],
        [3,  4.5,  24, False, False, 2],
        [3, 3.67,  24, False, False, 1],
        [5,    4,  40, True, True, 2],
        [5,    6,  40, True, True, 1],
        [5,    6,  40, True, True, 1],
        [5,    3,  48, True, True, 1],
        [5,    3,  48, True, True, 1],
        [5,    6,  96, True, True, 2],
        [5,    6,  96, True, True, 1],
        [5,    6,  96, True, True, 1]
    ]
    return MobileNetV3(cfgs, mode = 'small', **kwargs)
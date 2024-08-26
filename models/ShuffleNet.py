import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

def channel_shuffle(x, groups):
    b, c, h, w = x.data.size()
    channels_per_group = c // groups
    
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x

def conv1x1(in_channels, out_channels, groups = 1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 1,
        groups = groups,
        stride = 1
    )

def conv3x3(in_channels, out_channels, stride = 1, padding = 1, bias = True, groups = 1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = stride,
        padding = padding,
        bias = bias,
        groups = groups
    )

class ShuffleUnit(nn.Module):
    '''
    bottleneck channel의 크기: 초기 1x1 conv 수행 시 출력 채널의 1/4
    '''
    def __init__(self, in_channels, out_channels, groups = 3, grouped_conv = True, combine = 'add'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = self.out_channels // 4
        self.groups = groups
        self.grouped_conv = grouped_conv
        self.combine = combine
        
        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
        else:
            raise ValueError(f'Cannot combine tensors with "{self.combine}"' +
                             f'Only "add" and "concat" are supported.')
        
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm = True,
            relu = True
        )
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels,
            self.bottleneck_channels,
            stride = self.depthwise_stride,
            groups = self.bottleneck_channels
        )
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm = True,
            relu = False
        )
    
    @staticmethod
    def _add(x, out):
        return x + out
    
    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)
    
    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm = True, relu = False):
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups = groups)
        modules['conv1x1'] = conv
        
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv
    
    def forward(self, x):
        residual = x
        
        if self.combine == 'concat':
            residual = F.avg_pool2d(
                residual,
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)

class ShuffleNet(nn.Module):
    def __init__(self, groups = 3, in_channels = 3, num_classes = 1000):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.groups = groups
        self.stage_repeats = [3,7,3]
        
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(f"Only 1,2,3,4,8 is available!!!")
        
        self.conv1 = conv3x3(
            self.in_channels,
            self.stage_out_channels[1],
            stride = 2
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)
        
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)
        self._init_layer()
   
    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std = 0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = f"ShuffleUnit_Stage{stage}"
        
        grouped_conv = stage > 2
        
        first_module = ShuffleUnit(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage],
            groups = self.groups,
            grouped_conv = grouped_conv,
            combine = 'concat'
        )
        modules[stage_name + '_0'] = first_module
        
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + f"_{i+1}"
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups = self.groups,
                grouped_conv = True,
                combine = 'add'
            )
            modules[name] = module
        return nn.Sequential(modules)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, out.data.size()[-2:])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
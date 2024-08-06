import torch
from torch import nn, Tensor

class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = stride, padding = 1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace = True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace = True)
        )
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace = True)
        )
    
    def forward(self, x):
        out = self.block(x)
        return out
    
class MobileNetV1(nn.Module):
    def __init__(self, alpha, num_classes):
        '''
        alpha => width multiplier : 입력, 출력 채널을 alpha배 만큼 축소함!!
        ex) 출력 채널이 원래 64개이고, alpha = 0.25라면 축소된 출력 채널 개수는 16개
        -> 1,  0.75, 0.5, 0.25

        roh => resolution multiplier : 입력 영상 및 중간레이어들의 해상도를 roh배 만큼 축소함!!
        ex) 입력 영상의 해상도가 224X224였고, roh = 0.571이라면 축소된 해상도는 128X128이 됨!!
        -> 1(224), 0.857(192), 0.714(160), 0.571(128)

        총 16가지 조합을 실험함!
        '''
        super().__init__()
        self.conv1 = BasicConv2d(3, int(32 * alpha), kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = Depthwise(int(32 * alpha), int(64 * alpha), stride = 1)
        self.conv3 = nn.Sequential(
            Depthwise(int(64 * alpha), int(128 * alpha), stride = 2),
            Depthwise(int(128 * alpha), int(128 * alpha), stride = 1)
        )
        self.conv4 = nn.Sequential(
            Depthwise(int(128 * alpha), int(256 * alpha), stride = 2),
            Depthwise(int(256 * alpha), int(256 * alpha), stride = 1)
        )
        self.conv5 = nn.Sequential(
            Depthwise(int(256 * alpha), int(512 * alpha), stride = 2),
            *[Depthwise(int(512 * alpha), int(512 * alpha), stride = 1) for _ in range(5)]
        )
        self.conv6 = Depthwise(int(512 * alpha), int(1024 * alpha), stride = 2)
        self.conv7 = Depthwise(int(1024 * alpha), int(1024 * alpha), stride = 2)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
        
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
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.avg(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
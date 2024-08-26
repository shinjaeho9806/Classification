import torch
from torch import nn, Tensor

class Bottleneck(nn.Module):
    expansion_factor = 4
    def __init__(self, in_channels, inner_channels, cardinality = 32, base_width = 64, stride = 1):
        '''
        논문과 블로그 구현은 1번째 conv에 stride = stride, 2번째 conv에 stride = 1을 두지만,
        torchvision구현은 1번째 conv에 stride = 1, 2번째 conv에 stride = stride를 둠!
        '''
        super().__init__()
        # in_channels는 64
        # out_channels는 128
        # 128 = 64 * (4/64) * 32
        # 결과적으로 2배
        width = int(inner_channels * (base_width / 64.)) * cardinality
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size = 3, stride = stride, padding = 1, groups = cardinality, bias = False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, inner_channels * self.expansion_factor, kernel_size = 1, bias = False),
            nn.BatchNorm2d(inner_channels * self.expansion_factor)
        )
        if stride != 1 or in_channels != inner_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels * self.expansion_factor, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(inner_channels * self.expansion_factor),
            )
        else:
            self.residual = nn.Sequential()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        resid = x
        out = self.block(x)
        out += self.residual(resid)
        out = self.relu(out)
        return out

class ResNext_blueprint(nn.Module):
    def __init__(self, block, block_lst, cardinality = 32, base_width = 4, num_classes = 1000):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.in_channels = 64
        self.conv2 = self._make_layer(block, 64, block_lst[0], cardinality = cardinality, base_width = base_width, stride = 1)
        self.conv3 = self._make_layer(block, 128, block_lst[1], cardinality = cardinality, base_width = base_width, stride = 2)
        self.conv4 = self._make_layer(block, 256, block_lst[2], cardinality = cardinality, base_width = base_width, stride = 2)
        self.conv5 = self._make_layer(block, 512, block_lst[3], cardinality = cardinality, base_width = base_width, stride = 2)
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        
        self._init_layer()
    
    def _make_layer(self, block, inner_channels, num_blocks, cardinality, base_width, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, inner_channels, cardinality = cardinality, base_width = base_width, stride = stride))
            self.in_channels = inner_channels * block.expansion_factor
        return nn.Sequential(*layers)

    def _init_layer(self):
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
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avg(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out

class ResNeXt:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def ResNext50_32x4d(self):
        return ResNext_blueprint(Bottleneck, [3,4,6,3], cardinality = 32, base_width = 4)
    def ResNeXt101_32x8d(self):
        return ResNext_blueprint(Bottleneck, [3,4,23,3], cardinality = 32, base_width = 8)
    def ResNeXt101_64x4d(self):
        return ResNext_blueprint(Bottleneck, [3,4,23,3], cardinality = 64, base_width = 4)
    def ResNeXt152_32x4d(self):
        return ResNext_blueprint(Bottleneck, [3,8,36,3], cardinality = 32, base_width = 4)
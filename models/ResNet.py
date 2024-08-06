from torch import nn
import torch
from torch import Tensor

class Basicblock(nn.Module):
    expansion_factor = 1
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion_factor, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion_factor)
            )
        else:
            self.residual = nn.Sequential()
        
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        resid = x
        out = self.block(x)
        out += self.residual(resid)
        out = self.relu(out)
        return out

class BottleNeckblock(nn.Module):
    expansion_factor = 4
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels * self.expansion_factor, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels * self.expansion_factor),
        )
        
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion_factor, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion_factor)
            )
        else:
            self.residual = nn.Sequential()
            
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        resid = x
        out = self.block(x)
        out += self.residual(resid)
        out = self.relu(out)
        return out

class ResNet_blueprint(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        
        self.avg = nn.AdaptiveAvgPool2d(output_size = (1,1))
        
        self.fc = nn.Linear(512 * block.expansion_factor, num_classes)
        
        self._init_layer()
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)  
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion_factor
        return nn.Sequential(*layers)

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out
    
class ResNet:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def ResNet18(self):
        return ResNet_blueprint(Basicblock, [2,2,2,2], self.num_classes)
    
    def ResNet34(self):
        return ResNet_blueprint(Basicblock, [3,4,6,3], self.num_classes)
        
    def ResNet50(self):
        return ResNet_blueprint(BottleNeckblock, [3,4,6,3], self.num_classes)
        
    def ResNet101(self):
        return ResNet_blueprint(BottleNeckblock, [3,4,23,3], self.num_classes)
        
    def ResNet152(self):
        return ResNet_blueprint(BottleNeckblock, [3,8,36,3], self.num_classes)
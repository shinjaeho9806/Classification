import torch
from torch import nn, Tensor
from torchinfo import summary
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, eps = 0.001)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out, inplace = True)
    
class InceptionA(nn.Module):
    '''
    F5 -> A
    F5와 차이
    (1) branch2의 2번째 conv의 kernel_size : 3 -> 5, padding : 1 -> 2
    (2) branch4의 maxpool -> avgpool
    (3) pool_features옵션 추가
    '''
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(64, 96, kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(96, 96, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 48, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(48, 64, kernel_size = 5, stride = 1, padding = 2)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1),
            ConvBlock(in_channels, pool_features, kernel_size = 1, stride = 1, padding = 0)
        )
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size = 1, stride = 1, padding = 0)
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1,b2,b3,b4], dim = 1)
    
class InceptionC(nn.Module):
    '''
    F6 -> C
    F6와 C의 차이
    (1) branch1의 (1xn)>(nx1)이 (nx1)>(1xn)으로 변경됨
    '''
    def __init__(self, in_channels, f_7x7):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(f_7x7, f_7x7, kernel_size = (7,1), stride = 1, padding = (3,0)),
            ConvBlock(f_7x7, f_7x7, kernel_size = (1,7), stride = 1, padding = (0,3)),
            ConvBlock(f_7x7, f_7x7, kernel_size = (7,1), stride = 1, padding = (3,0)),
            ConvBlock(f_7x7, 192, kernel_size = (1,7), stride = 1, padding = (0,3))
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(f_7x7, f_7x7, kernel_size = (1,7), stride = 1, padding = (0,3)),
            ConvBlock(f_7x7, 192, kernel_size = (7,1), stride = 1, padding = (3,0))
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride = 1, padding = 1),
            ConvBlock(in_channels, 192, kernel_size = 1, stride = 1, padding = 0)
        )
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size = 1, stride = 1, padding = 0)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim = 1)

class InceptionE(nn.Module):
    '''
    F7 -> E
    F7과 E차이점
    (1) branch3의 maxpool -> avgpool
    '''
    def __init__(self, in_channels):
        super().__init__()
        # branch1
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 448, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(448, 384, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch1_width = ConvBlock(384, 384, kernel_size = (1,3), stride = 1, padding = (0,1))
        self.branch1_height = ConvBlock(384, 384, kernel_size = (3,1), stride = 1, padding = (1,0))
        # branch2
        self.branch2 = ConvBlock(in_channels, 384, kernel_size = 1, stride = 1, padding = 0)
        self.branch2_width = ConvBlock(384, 384, kernel_size = (1,3), stride = 1, padding = (0,1))
        self.branch2_height = ConvBlock(384, 384, kernel_size = (3,1), stride = 1, padding = (1,0))
        # branch3
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1),
            ConvBlock(in_channels, 192, kernel_size = 1, stride = 1, padding = 0)
        )
        # branch4
        self.branch4 = ConvBlock(in_channels, 320, kernel_size = 1, stride = 1, padding = 0)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b1_w = self.branch1_width(b1)
        b1_h = self.branch1_height(b1)
        b2 = self.branch2(x)
        b2_w = self.branch2_width(b2)
        b2_h = self.branch2_height(b2)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1_w, b1_h, b2_w, b2_h, b3, b4], dim = 1)

class InceptionB(nn.Module):
    '''
    Red -> B
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(64, 96, kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(96, 96, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 384, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch3 = nn.MaxPool2d(3, stride = 2, padding = 0)
        
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim = 1)

class InceptionD(nn.Module):
    '''
    V3에 새로 추가된 레이어
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(192, 192, kernel_size = (1,7), stride = 1, padding = (0,3)),
            ConvBlock(192, 192, kernel_size = (7,1), stride = 1, padding = (3,0)),
            ConvBlock(192, 192, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(192, 320, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch3 = nn.MaxPool2d(3, stride = 2, padding = 0)
        
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim = 1)

class InceptionAux(nn.Module):
    '''
    V2와 차이
    (1) block구조 변경
    (2) adaptive avg를 블록다음에 수행
    (3) linear를 하나로 축소
    '''
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            ConvBlock(in_channels, 128, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(128, 768, kernel_size = 5, stride = 1, padding = 0)
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        
    def forward(self,x):
        out = self.block(x)
        out = self.avg(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class InceptionV3(nn.Module):
    def __init__(self, aux_logits = True, num_classes = 10):
        super().__init__()
        assert aux_logits == True or aux_logits == False

        self.aux_logits = aux_logits
        
        self.conv1 = ConvBlock(3, 32, kernel_size = 3, stride = 2, padding = 0)
        self.conv2 = ConvBlock(32, 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv3 = ConvBlock(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool1 = nn.MaxPool2d(3, stride = 2, padding = 0)
        self.conv4 = ConvBlock(64, 80, kernel_size = 1, stride = 1, padding = 0)
        self.conv5 = ConvBlock(80, 192, kernel_size = 3, stride = 1, padding = 0)
        self.maxpool2 = nn.MaxPool2d(3, stride = 2, padding = 0)
        
        self.inception3a = InceptionA(192, pool_features = 32)
        self.inception3b = InceptionA(256, pool_features = 64)
        self.inception3c = InceptionA(288, pool_features = 64)
        
        self.inceptionRed1 = InceptionB(288)
        
        self.inception4a = InceptionC(768, f_7x7 = 128)
        self.inception4b = InceptionC(768, f_7x7 = 160)
        self.inception4c = InceptionC(768, f_7x7 = 160)
        self.inception4d = InceptionC(768, f_7x7 = 192)
        
        if self.aux_logits:
            self.aux = InceptionAux(768, num_classes)
        else:
            self.aux = None

        self.inceptionRed2 = InceptionD(768)
        
        self.inception5a = InceptionE(1280)
        self.inception5b = InceptionE(2048)
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, num_classes)
        
        self._init_layer()
    
    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool1(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.inception3c(out)
        out = self.inceptionRed1(out)

        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        
        if self.aux_logits and self.training:
            aux = self.aux(out)
        out = self.inceptionRed2(out)
        
        out = self.inception5a(out)
        out = self.inception5b(out)
        
        out = self.avg(out)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if self.aux_logits and self.training:
            return out, aux
        else:
            return out
import torch
from torch import nn, Tensor
from torchinfo import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.block(x)
    
class InceptionF5(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(64, 96, kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(96, 96, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 48, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(48, 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride = 1, padding = 1),
            ConvBlock(in_channels, 64, kernel_size = 1, stride = 1, padding = 0)
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

class InceptionF6(nn.Module):
    def __init__(self, in_channels, f_7x7):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(f_7x7, f_7x7, kernel_size = (1,7), stride = 1, padding = (0,3)),
            ConvBlock(f_7x7, f_7x7, kernel_size = (7,1), stride = 1, padding = (3,0)),
            ConvBlock(f_7x7, f_7x7, kernel_size = (1,7), stride = 1, padding = (0,3)),
            ConvBlock(f_7x7, 192, kernel_size = (7,1), stride = 1, padding = (3,0))
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
    
class InceptionF7(nn.Module):
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
            nn.MaxPool2d(3, stride = 1, padding = 1),
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

class InceptionRed(nn.Module):
    def __init__(self, in_channels, f_3x3_r, add_ch = 0):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(f_3x3_r, 178 + add_ch, kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(178 + add_ch, 178 + add_ch, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(f_3x3_r, 302 + add_ch, kernel_size = 3, stride = 2, padding = 0)
        )
        self.branch3 = nn.MaxPool2d(3, stride = 2, padding = 0)
        
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim = 1)
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Conv2d(in_channels, 128, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(inplace = True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self,x):
        out = self.block(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class InceptionV2(nn.Module):
    def __init__(self, aux_logits = True, num_classes = 10):
        super().__init__()
        assert aux_logits == True or aux_logits == False

        self.aux_logits = aux_logits
        
        self.conv1 = ConvBlock(3, 32, kernel_size = 3, stride = 2, padding = 0)
        self.conv2 = ConvBlock(32, 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv3 = ConvBlock(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool1 = nn.MaxPool2d(3, stride = 2, padding = 0)
        self.conv4 = ConvBlock(64, 80, kernel_size = 3, stride = 1, padding = 0)
        self.conv5 = ConvBlock(80, 192, kernel_size = 3, stride = 2, padding = 0)
        self.conv6 = ConvBlock(192, 288, kernel_size = 3, stride = 1, padding = 1)
        
        self.inception3a = InceptionF5(288)
        self.inception3b = InceptionF5(288)
        self.inception3c = InceptionF5(288)
        
        self.inceptionRed1 = InceptionRed(288, f_3x3_r = 64, add_ch = 0)
        
        self.inception4a = InceptionF6(768, f_7x7 = 128)
        self.inception4b = InceptionF6(768, f_7x7 = 160)
        self.inception4c = InceptionF6(768, f_7x7 = 160)
        self.inception4d = InceptionF6(768, f_7x7 = 160)
        self.inception4e = InceptionF6(768, f_7x7 = 192)
        
        self.inceptionRed2 = InceptionRed(768, f_3x3_r = 192, add_ch = 16)
        
        if self.aux_logits:
            self.aux = InceptionAux(768, num_classes)
        else:
            self.aux = None
        
        self.inception5a = InceptionF7(1280)
        self.inception5b = InceptionF7(2048)
        
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
        out = self.conv6(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.inception3c(out)
        out = self.inceptionRed1(out)

        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        
        
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
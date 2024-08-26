import torch
from torch import nn, Tensor

def ConvBlock(in_channels, out_channels, **kwargs):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, n1x1, kernel_size = 1, stride = 1, padding = 0)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(n3x3_reduce, n3x3, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(n5x5_reduce, n5x5, kernel_size = 5, stride = 1, padding = 2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(in_channels, pool_proj, kernel_size = 1, stride = 1, padding = 0)
        )
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = torch.cat([x1,x2,x3,x4], dim = 1)
        return out

class Auciliary_classifier(nn.Module):
    '''
    Linear 입력 : 128 * 4 * 4
    '''
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels, 128, kernel_size = 1, stride = 1, padding = 0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        out = self.block(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class GoogleNet(nn.Module):
    '''
    ImageNet기준 -> (3,224,224)
    '''
    def __init__(self, aux_logits = True, num_classes = 1000):
        super().__init__()
        assert aux_logits == True or aux_logits == False
        
        self.aux_logits = aux_logits
        
        self.front_block = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),   # (64, 112, 112)
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),                                    # (64, 56, 56)
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),  # (64, 56, 56)
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1, padding = 1), # (192, 56, 56)
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)                                     # (192, 28, 28)
        )
        
        self.inception_a3 = Inception(192, 64, 96, 128, 16, 32, 32)                                     # (256, 28, 28) out_channels -> 256 = 64 + 128 + 32 + 32
        self.inception_b3 = Inception(256, 128, 128, 192, 32, 96, 64)                                   # (480, 28, 28) out_channels -> 480 = 128 + 192 + 96 + 64
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)                          # (480, 14, 14)
        self.inception_a4 = Inception(480, 192, 96, 208, 16, 48, 64)                                    # (512, 14, 14) out_channels -> 512 = 192 + 208 + 48 + 64
        self.inception_b4 = Inception(512, 160, 112, 224, 24, 64, 64)                                   # (512, 14, 14) out_channels -> 512 = 160 + 224 + 64 + 64
        self.inception_c4 = Inception(512, 128, 128, 256, 24, 64, 64)                                   # (512, 14, 14) out_channenls -> 512 =  128 + 256 + 64 + 64
        self.inception_d4 = Inception(512, 112, 144, 288, 32, 64, 64)                                   # (538, 14, 14) out_channels -> 528 = 112 + 288 + 64 + 64
        self.inception_e4 = Inception(528, 256, 160, 320, 32, 128, 128)                                 # (832, 14, 14) out_channels ->  832 = 256 + 320 + 128 + 128
        self.maxpool4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)                          # (832, 7, 7)
        self.inception_a5 = Inception(832, 256, 160, 320, 32, 128, 128)                                 # (832, 7, 7) out_channels ->  832 = 256 + 320 + 128 + 128
        self.inception_b5 = Inception(832, 384, 192, 384, 48, 128, 128)                                 # (1024, 7, 7) out_channels ->  1024 = 384 + 384 + 128 + 128
        self.avg = nn.AvgPool2d(kernel_size = 7, stride = 1)                                            # (1024, 1, 1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)                                                          # (1000)
        
        if self.aux_logits:
            self.aux1 = Auciliary_classifier(512, num_classes)
            self.aux2 = Auciliary_classifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
    
    def _init_layer(self):
        for m in self.module():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.front_block(x)
        out = self.inception_a3(out)
        out = self.inception_b3(out)
        out = self.maxpool3(out)
        out = self.inception_a4(out)
        if self.aux_logits and self.training:
            aux1 = self.aux1(out)
        out = self.inception_b4(out)
        out = self.inception_c4(out)
        out = self.inception_d4(out)
        if self.aux_logits and self.training:
            aux2 = self.aux2(out)
        out = self.inception_e4(out)
        out = self.maxpool4(out)
        out = self.inception_a5(out)
        out = self.inception_b5(out)
        out = self.avg(out)
        out = torch.flatten(out,1)
        out = self.dropout(out)
        out = self.fc(out)
        if self.aux_logits and self.training:
            return out, aux1, aux2
        else:
            return out
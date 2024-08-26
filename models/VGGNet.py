import torch
from torch import nn

def conv_block(num_layer, in_channels, out_channels):
    if num_layer == '1':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        )
    if num_layer == '2':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        )
    elif num_layer == '3_1':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        )
    elif num_layer == '3':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        )
    elif num_layer == '4':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        )
    return block

class VGG_blueprint(nn.Module):
    '''
    ### Linear입력 ###
    32 X 32 : 512 * 1 * 1 -> fc_features = 1
    224 X 224 : 512 * 7 * 7 -> fc_features = 7
    512 X 512 : 512 * 32 * 32 -> fc_features = 32
    '''
    def __init__(self, model_type, dropout_p, fc_features, num_classes):
        super().__init__()
        self.model_type = model_type
        self.dropout_p = dropout_p

        if model_type == 'A':
            self.backbone = nn.Sequential(
                conv_block('1',3,64),
                conv_block('1',64,128),
                conv_block('2',128,256),
                conv_block('2',256,512),
                conv_block('2',512,512)
            )
        elif model_type == 'B':
            self.backbone = nn.Sequential(
                conv_block('2',3,64),
                conv_block('2',64,128),
                conv_block('2',128,256),
                conv_block('2',256,512),
                conv_block('2',512,512)
            )
        elif model_type == 'C':
            self.backbone = nn.Sequential(
                conv_block('2',3,64),
                conv_block('2',64,128),
                conv_block('3_1',128,256),
                conv_block('3_1',256,512),
                conv_block('3_1',512,512)
            )
        elif model_type == 'D':
            self.backbone = nn.Sequential(
                conv_block('2',3,64),
                conv_block('2',64,128),
                conv_block('3',128,256),
                conv_block('3',256,512),
                conv_block('3',512,512)
            )
        elif model_type == 'E':
            self.backbone = nn.Sequential(
                conv_block('2',3,64),
                conv_block('2',64,128),
                conv_block('4',128,256),
                conv_block('4',256,512),
                conv_block('4',512,512)
            )
        else:
            raise Exception('Please select in A,B,C,D,E !!!')
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * fc_features * fc_features, 4096),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(4096,num_classes)
        )
        self._init_layer()
        
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
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
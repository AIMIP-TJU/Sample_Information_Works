import torch.nn as nn
import math
import torch.nn.functional as F

class VGG(nn.Module):

    def __init__(self, features, num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(          
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),          
        )
        self.linear2 = nn.Linear(512, num_classes)        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)        
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.classifier(x)
        embedding = x.detach()        
        x = self.linear2(x)
        x = F.relu(x)
        #print(type(x), type(embedding))
        return x, embedding

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, num_input, batch_norm=False):
    layers = []
    in_channels = num_input
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_input, num_classes):
    model = VGG(make_layers(cfg['A'], num_input), num_classes)
    return model


def vgg11_bn(num_input, num_classes):
    model = VGG(make_layers(cfg['A'], num_input, batch_norm=True), num_classes)
    return model


def vgg13(num_input, num_classes):
    model = VGG(make_layers(cfg['B'], num_input), num_classes)
    return model


def vgg13_bn(num_input, num_classes):
    model = VGG(make_layers(cfg['B'], num_input, batch_norm=True), num_classes)
    return model


def vgg16(num_input, num_classes):
    model = VGG(make_layers(cfg['D'], num_input), num_classes)
    return model


def vgg16_bn(num_input, num_classes):
    model = VGG(make_layers(cfg['D'], num_input, batch_norm=True), num_classes)
    return model


def vgg19(num_input, num_classes):
    model = VGG(make_layers(cfg['E'], num_input), num_classes)
    return model


def vgg19_bn(num_input, num_classes):
    model = VGG(make_layers(cfg['E'], num_input, batch_norm=True), num_classes)
    return model


if __name__ == '__main__':
    # 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
    # Example
    net11 = vgg11(3, 10)
    print(net11)

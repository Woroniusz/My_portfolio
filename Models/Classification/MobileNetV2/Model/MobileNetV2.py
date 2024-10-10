import torch
from torch import nn

class InvertedResidual(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, stride: int, expansion: int) -> None:
        super(InvertedResidual, self).__init__()
        hidden = inChannels*expansion
        self.useConnect = inChannels == outChannels and stride == 1
        
        layers = nn.ModuleList()
        
        if expansion != 1:
            layers.append(nn.Conv2d(inChannels, hidden, (1,1), stride=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(hidden, hidden, (3,3), stride=stride, padding=1, groups=hidden, bias=False))
        layers.append(nn.BatchNorm2d(hidden))
        layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(hidden, outChannels, (1,1), stride=1, bias=False))
        layers.append(nn.BatchNorm2d(outChannels))

        self.sequential = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        inner = self.sequential(x)
        if self.useConnect:
            return x + inner
        else:
            return inner
        
        
class MobileNetV2(nn.Module):
    def __init__(self, numClasses = 1000, dropout = 0.4):
        super(MobileNetV2, self).__init__()
        
        
        
        self.firstLayer = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        cfg = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        
        layers = nn.ModuleList()
        
        inDim = 32
        
        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(inDim, c, stride, t))
                inDim = c
        
        layers.append(nn.Conv2d(320, 1280, (1,1), stride=1, bias=False))
        layers.append(nn.BatchNorm2d(1280))
        layers.append(nn.ReLU6(inplace=True))
        
        self.secLayers = nn.Sequential(*layers)
        
        self.lastLayer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, numClasses)
        )
        self.avp = nn.AvgPool2d((7,7))
        
    def forward(self, x):
        x = self.firstLayer(x)
        x = self.secLayers(x)
        x = self.avp(x)
        x = x.view(x.size(0), -1)
        return self.lastLayer(x)
    


if __name__ == "__main__":
    input = torch.rand(1, 3, 224, 224)
    model = MobileNetV2()
    output = model(input)
    print(output.shape)
        
        
        
        
        
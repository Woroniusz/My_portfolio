import torch
from torch import nn
import numpy as np



class UpSample(nn.Module):
    def __init__(self, size: (int,int), inDim: int, outDim: int, dropOut: float = 0.5) -> None:
        super(UpSample, self).__init__()
        self.seq = nn.Sequential(
            nn.UpsamplingBilinear2d(size=size),
            nn.Conv2d(inDim, outDim, 3, 1, 1),
            nn.Dropout(dropOut),
            nn.BatchNorm2d(outDim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x: torch.Tensor):
        return self.seq(x)

class Generator(nn.Module):
    def __init__(self, inputDim: int = 128, outputDim: int = 1, outputSize: (int,int) = (28,28), dropout: float = 0.5) -> None:
        """
        Parameter:
            inputDim:       Dimension of input noise (64)
            outputsize:     Size of output image (28,28)
            outputDim:      Dimension of OutImage (1)
        Input:
            x [B x inputDim]: noise
        Output:
            Image [B x outputDim x outputsize]: Generated image
        """
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Tanh()
        )
        
        
        
    def forward(self, x: torch.Tensor):
        return self.seq(x).view(-1, 1, 28, 28)
    
    
    
if __name__ == "__main__":
    input = torch.rand(1, 128)
    generator = Generator()
    output = generator(input)
    print(output.shape)
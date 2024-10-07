import torch
from torch import nn




class Discriminator(nn.Module):
    def __init__(self, dimInput = 1, dropOut = 0.5, sizeOfInput = (28,28)) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x: torch.Tensor):
        return self.seq(x)
    
    
if __name__ == "__main__":
    input = torch.rand(1, 1, 28, 28)
    discriminator = Discriminator()
    output = discriminator(input)
    print(output.shape)
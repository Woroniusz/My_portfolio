import torch
from torch import nn
from Model.Discriminator import Discriminator
from Model.Generator import Generator
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import argparse
import tqdm
from torchvision.utils import save_image
import numpy as np
import cv2
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def GetArg(arg: argparse.ArgumentParser):
    arg.add_argument("--batch", default=64)
    arg.add_argument("--epoch", default=500),
    arg.add_argument("--mini", default=8)
    arg.add_argument("--weight", default="")
    arg.add_argument("--lr", default=1e-5)
    return arg.parse_args()



if __name__ == "__main__":
    arg = GetArg(argparse.ArgumentParser())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    discriminator = Discriminator()
    generator = Generator()
    if arg.weight:
        discriminator.load_state_dict(torch.load(f"{arg.weight}/disc.pth", map_location=device))
        generator.load_state_dict(torch.load(f"{arg.weight}/gen.pth", map_location=device))
        
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    dataset = MNIST(root=".", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ]))
    dataLoad = DataLoader(dataset, arg.batch, True)
    cross = nn.BCEWithLogitsLoss()
    
    optDiscri = torch.optim.AdamW(discriminator.parameters(), arg.lr)
    optGenerator = torch.optim.AdamW(generator.parameters(), arg.lr)

    
    
    
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    if not os.path.exists(f"save"):
        try:
            os.makedirs("save")
            print("PAth has been created")
        except Exception as e:
            print("Path hasn't been created")
            print(f"Error Code: {e}")
            exit(0)
    
    path = f"save/{now}"
    os.makedirs(path)
    
    summer = SummaryWriter(path)
    
    step = 0
    for i in tqdm.tqdm(range(arg.epoch)):
        DiscriminatorLoss = []
        GeneratorLoss = []
        for j, (img, label) in enumerate(dataLoad):
            
            valid = torch.ones(img.size(0), 1, requires_grad=False).to(device)
            fake = torch.zeros(img.size(0), 1, requires_grad=False).to(device)
            
            optGenerator.zero_grad()
            z = torch.from_numpy(np.random.normal(0, 1, (img.shape[0], 128))).float().to(device)
            
            genImage = generator(z)
            GenLoss = cross(discriminator(genImage), valid)
            
            GenLoss.backward()
            optGenerator.step()
            GeneratorLoss.append(GenLoss.item())
            realImg = img.to(device)
            
            optDiscri.zero_grad()
            
            realOut = discriminator(realImg)
            fakeOut = discriminator(genImage.detach())
            
            realLoss = cross(realOut, valid)
            fakeLoss = cross(fakeOut, fake)
            
            loss = (realLoss + fakeLoss)/2
            loss.backward()
            optDiscri.step()
            DiscriminatorLoss.append(loss.item())
            if j % 400 == 0:
                summer.add_scalars("loss", {
                    "Generator":  np.array(GeneratorLoss).mean(),
                    "Discriminator": np.array(DiscriminatorLoss).mean()
                },
                step)
                Image = np.zeros((1, 28*5, 28*5))
                generateImage = genImage[:25].cpu().detach().numpy()
                for x in range(5):
                    for y in range(5):
                        Image[:, x*28:28+x*28, y*28:28+y*28] = generateImage[x*5 + y]
                summer.add_image("image", Image, step)
                step += 1
            torch.save(discriminator.state_dict(), f"{path}/disc.pth")
            torch.save(generator.state_dict(), f"{path}/gen.pth")
            
            
            
    
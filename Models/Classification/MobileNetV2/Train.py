import torch
import os
from Model.MobileNetV2 import MobileNetV2
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def GetArgs(arg: argparse.ArgumentParser):
    arg.add_argument("--batch", default=32, type=int)
    arg.add_argument("--epoch", default=300, type=int)
    arg.add_argument("--lr", default=1e-5, type=float)
    arg.add_argument("--save", default="save", type=str)
    arg.add_argument("--weight", default="", type=str)
    return arg.parse_args()


def main(arg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MobileNetV2(numClasses=100)
    if arg.weight:
        model.load_state_dict(torch.load(arg.weight, map_location=device))
    model = model.to(device)
    
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x/255.),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainData = CIFAR100(".", train=True, download=True, transform=transform)
    validnData = CIFAR100(".", train=False, download=True, transform=transform)
    
    trainData = DataLoader(trainData, arg.batch, True)
    validnData = DataLoader(validnData, arg.batch, True)
    
    
    
    optimalizer = torch.optim.AdamW(model.parameters(), lr=arg.lr)
    lossFunction = torch.nn.CrossEntropyLoss()
    
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    pathSave = f"{arg.save}/{now}"
    os.makedirs(pathSave)
    writter = SummaryWriter(pathSave)
    
    
    F1 = 100
    for epoch in range(arg.epoch):
        lossArrayTrain = []
        yPredTrain = []
        yLabelTrain = []
        
        
        print(f"""
              =========Train===========
              ===== Epoch: {epoch}=====
              """)
        
        model.train()
        for img, label in tqdm(trainData):
            img = img.to(device)
            label = label.to(device)
            optimalizer.zero_grad()
            out = model(img)
            loss = lossFunction(out, label)
            loss.backward()
            optimalizer.step()
            lossArrayTrain.append(loss.item())
            yPredTrain.append(out.detach().cpu().argmax(1).numpy())
            yLabelTrain.append(label.cpu().numpy())
        
        yPredTrain = np.concatenate(yPredTrain)
        yLabelTrain = np.concatenate(yLabelTrain)
        
        model.eval()
        
        lossArrayValid = []
        yPredValid = []
        yLabelValid = []
        print(f"""
              =========Valid===========
              ===== Epoch: {epoch}=====
              """)
        with torch.no_grad():
            for img, label in tqdm(validnData):
                img = img.to(device)
                label = label.to(device)
                
                out = model(img)
                loss = lossFunction(out, label)

                lossArrayValid.append(loss.item())
                yPredValid.append(out.detach().cpu().argmax(1).numpy())
                yLabelValid.append(label.cpu().numpy())    
                
        
        yPredValid = np.concatenate(yPredValid)
        yLabelValid = np.concatenate(yLabelValid)
        
        accuraceTrain = accuracy_score(y_true=yLabelTrain, y_pred=yPredTrain)
        accuraceValid = accuracy_score(y_true=yPredValid, y_pred=yLabelValid)
        
        precisionTrain = precision_score(y_true=yLabelTrain, y_pred=yPredTrain, average="micro")
        precisionValid = precision_score(y_true=yPredValid, y_pred=yLabelValid, average="micro")
        
        recallTrain = recall_score(y_true=yLabelTrain, y_pred=yPredTrain, average="micro")
        recallValid = recall_score(y_true=yPredValid, y_pred=yLabelValid, average="micro")
        
        f1Train = f1_score(y_true=yLabelTrain, y_pred=yPredTrain, average="micro")
        f1Valid = f1_score(y_true=yPredValid, y_pred=yLabelValid, average="micro")
        
        
        writter.add_scalars("loss", {"Train": np.array(lossArrayTrain).mean(),
                                        "Valid": np.array(lossArrayValid).mean()}, epoch)
        
        writter.add_scalars("Accurace", {"Train": accuraceTrain,
                                            "Valid": accuraceValid}, epoch)
        
        writter.add_scalars("Precision", {"Train": precisionTrain,
                                            "Valid": precisionValid}, epoch)
        
        writter.add_scalars("Recall", {"Train": recallTrain,
                                        "Valid": recallValid}, epoch)
        
        writter.add_scalars("F1", {"Train": f1Train,
                                    "Valid": f1Valid}, epoch)
        
        
        print(f"""
                TRAIN:
                    -loss:      |   {np.array(lossArrayTrain).mean()}
                    -Accurace:  |   {accuraceTrain}
                    -Precision: |   {precisionTrain},
                    -Recall:    |   {recallTrain}
                    -F1:        |   {f1Train}
                ========================================================
                Valid:
                    -loss:      |   {np.array(lossArrayValid).mean()}
                    -Accurace:  |   {accuraceValid}
                    -Precision: |   {precisionValid},
                    -Recall:    |   {recallValid}
                    -F1:        |   {f1Valid}
                """)
        
        if F1 > f1Valid:
            F1 = f1Valid
            torch.save(model.state_dict(), f"{pathSave}/best.pth")
        torch.save(model.state_dict(), f"{pathSave}/last.pth")
        

if __name__ == "__main__":
    main(GetArgs(argparse.ArgumentParser()))

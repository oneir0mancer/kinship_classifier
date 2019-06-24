import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from glob import glob
from collections import defaultdict

#Get all images and split them into train and validation set
print("Prepare data...")
train_file_path = "data/train_relationships.csv"
train_folders_path = "data/train/"
val_famillies = "F09"

all_images = glob(train_folders_path + "*/*/*.jpg")

train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
	
relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train_relations = [x for x in relationships if val_famillies not in x[0]]
val_relations  = [x for x in relationships if val_famillies in x[0]]

#Prepate dataloaders
from .dataset import KinDataset
from .model import SiameseNet

from torch.utils.data import DataLoader
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) 
])
val_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) 
])

trainset = KinDataset(train_relations, train_person_to_images_map, train_transform)
valset = KinDataset(val_relations, val_person_to_images_map, val_transform)

trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
valloader = DataLoader(valset, batch_size=200, shuffle=False)

print("Initialize network...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SiameseNet().to(device)

lr = 1e-3

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, patience=10)

def train():
    net.train()
    train_loss = 0.0
    running_loss = 0.0
    running_corrects = 0
    
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1,1).to(device)
        output = net(img1, img2)
        preds = output>0.5
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == (label>0.5))
        
        step = 100
        if i % step == step-1:
            print(' [{} - {:.2f}%],\ttrain loss: {:.5}'.format(epoch+1, 100*(i+1)/len(trainloader), running_loss/step/200))
            running_loss = 0
        
    train_loss /= len(trainset)
    running_corrects = running_corrects.item()/len(trainset)
    print('[{}], \ttrain loss: {:.5}\tacc: {:.5}'.format(epoch+1, train_loss, running_corrects))
    return train_loss, running_corrects
  
def validate():
    net.eval()
    val_loss = 0.0
    running_corrects = 0
    
    for batch in valloader:
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1,1).to(device)
        with torch.no_grad():
            output = net(img1, img2)
            preds = output>0.5
            loss = criterion(output, label)
            
        val_loss += loss.item()
        running_corrects += torch.sum(preds == (label>0.5))
    
    val_loss /= len(valset)
    running_corrects = running_corrects.item()/len(valset)
    print('[{}], \tval loss: {:.5}\tacc: {:.5}'.format(epoch+1, val_loss, running_corrects))
    	
    return val_loss, running_corrects
	

print("Start training...")	
num_epoch = 100

best_val_loss = 1000
best_epoch = 0

history = []
accuracy = []
for epoch in range(num_epoch):
    train_loss, train_acc = train()  
    val_loss, val_acc = validate()
    history.append((train_loss, val_loss))
    accuracy.append((train_acc,val_acc))
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.torch.save(net.state_dict(), 'net_checkpoint.pth')
		
torch.save(net.state_dict(), 'net_full_training.pth')

#%pylab inline
#plt.plot([x[0] for x in history], 'b', label='train')
#plt.plot([x[1] for x in history], 'r--',label='validation')
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend()
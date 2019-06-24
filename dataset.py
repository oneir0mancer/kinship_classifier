import torchvision
from random import choice
from torch.utils.data import Dataset
from PIL import Image

class KinDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = list(person_to_images_map.keys())

    def __len__(self):
        return len(self.relations)*2
               
    def __getitem__(self, idx):
        
        if (idx%2==0): #Positive samples
            p1, p2 = self.relations[idx//2]
            label = 1
        else:          #Negative samples
            while True:
                p1 = choice(self.ppl)
                p2 = choice(self.ppl)
                if p1 != p2 and (p1, p2) not in self.relations and (p2, p1) not in self.relations:
                    break 
            label = 0
            
        path1, path2 = choice(self.person_to_images_map[p1]), choice(self.person_to_images_map[p2])
        img1, img2 = Image.open(path1), Image.open(path2)
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        
        return img1, img2, label
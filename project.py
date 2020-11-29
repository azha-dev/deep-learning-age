import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FaceAgesDataset import FaceAgesDataset

FILTERS_2 = 16
KERNEL_2 = 5

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolution = nn.Conv2d(3, 6, 5) #Epaisseur d'entrée couleur, le nombre de filtres qu'on veut, la taille impaire des kernels (5x5)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.convolution2 = nn.Conv2d(6, FILTERS_2, KERNEL_2) #Nombre de filtres de la 1ère conv, nouveau nombre de filtres
        self.pooling2 = nn.MaxPool2d(2, 2) 
        self.layer1  = nn.Linear(35344, 100) #Le plus dur à calculer, c'est le 5x5x6 (prise de tête, regarder la doc avec la formule)
        self.layer2  = nn.Linear(100, 50)
        self.layer3  = nn.Linear(50, 10)
        self.layer4  = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.pooling1(torch.nn.functional.relu(self.convolution(x)))
        x = self.pooling2(torch.nn.functional.relu(self.convolution2(x)))
        x = x.view(-1, 35344) #Convertit le tenseur en vecteur
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        return x
    
if __name__ == "__main__":
    transform = transforms.ToTensor()
    ages_dataset = FaceAgesDataset('images/inputs.csv', 'images/', transform)
   
    loader = torch.utils.data.DataLoader(ages_dataset)

    print(len(ages_dataset))
    img, label = ages_dataset[0]
    print(img.size(), label)

    #npimg = img.numpy()
    #npimg = np.transpose(npimg, (1, 2, 0))
    #plt.imshow(npimg)
    #plt.show()

    cnn = CNN()

    optimizer = torch.optim.SGD(cnn.parameters(), lr=1e-4)
    errorFunction = nn.MSELoss()


    for iteration in range(1):
        for i, img in enumerate(loader):
            inputs, label = img
            
            optimizer.zero_grad() #Initialisation de l'optimiseur
            output = cnn(inputs)
            label = label.type(torch.FloatTensor)
            error = errorFunction(output, label)
            error.backward()
            optimizer.step()

            if i % 100 == 0:
                print(error)

    torch.save(cnn.state_dict(), "project.dat") #Enregistrement du réseau dans un fichier

    #Pour ouvrir dans un autre script
    cnn = CNN()
    cnn.load_state_dict(torch.load("project.dat"))
    cnn.eval() #toujours commencer par ça pour construire le réseau

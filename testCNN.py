from project import CNN 
from FaceAgesDataset import FaceAgesDataset
import torch
import torchvision
import torchvision.transforms as transforms

cnn = CNN()
cnn.load_state_dict(torch.load("project.dat"))
cnn.eval() #toujours commencer par ça pour construire le réseau


transform = transforms.ToTensor()
test_ages_dataset = FaceAgesDataset('images-test/inputs.csv', 'images-test/', transform)

test_loader = torch.utils.data.DataLoader(test_ages_dataset)
correct = 0
total = 0
for images, labels in test_loader:
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))
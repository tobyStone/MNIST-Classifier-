import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_path = './'
#set up transforms
transform = transforms.Compose([transforms.ToTensor()])

# Load train and test datasets


training_data = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform, download=True)

 
test_data = torchvision.datasets.MNIST(root=image_path, train=False, transform=transform, download=True)

validation_set, test_set = torch.utils.data.random_split(test_data, [int(0.8 * len(test_data,)), int(0.2 * len(test_data))])

# Create the training and test dataloaders with a batch size of 64
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
val_loader  = DataLoader(validation_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)



def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(int(labels[i].detach()))
    
        image = images[i].numpy()
        plt.imshow(image.T.squeeze().T)
        plt.show()

# Explore data
figure = show5(train_loader)

# Define the class for your neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.convolution = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.max_pool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)
        self.dropout = nn.Dropout (p=0.5)

    def forward(self, x):
#        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.convolution(x)
        x = self.activation(x)
        x = self.max_pool2d(x)
        x = self.convolution2(x)
        x = self.activation(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the model
net = Net()
net.to(device)



# Choose an optimizer

optimizer = optim.Adam(net.parameters(), lr=0.001)

# Choose a loss function
criterion = nn.CrossEntropyLoss()

num_epochs = 10

# Establish a list for our history
train_loss_history = list()
val_loss_history = list()

for epoch in range(num_epochs):
    if torch.cuda.is_available():
        net = net.to('cuda')
    net.train()
    train_loss = 0.0
    train_correct = 0
    for i, data in enumerate(train_loader):
        # data is a list of [inputs, labels]
        inputs, labels = data

        # Pass to GPU if available.
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()



        _, preds = torch.max(outputs.data, 1)
        train_correct += (preds == labels).sum().item()
        train_loss += loss.item()
    print(f'Epoch {epoch + 1} training accuracy: {train_correct/len(train_loader):.2f}% training loss: {train_loss/len(train_loader):.5f}')
    train_loss_history.append(train_loss/len(train_loader))
    
    val_loss = 0.0
    val_correct = 0
    net.eval()
 
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        val_correct += (preds == labels).sum().item()
        val_loss += loss.item()
    print(f'Epoch {epoch + 1} val accuracy: {val_correct/len(val_loader):.2f}% validation loss: {val_loss/len(val_loader):.5f}')
    val_loss_history.append(val_loss/len(val_loader))




       
plt.plot(train_loss_history, label="Training Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.legend()
plt.show()


net.eval()
results = list()
total = 0
for itr, (image, label) in enumerate(test_loader):

    if (torch.cuda.is_available()):
        image = image.cuda()
        label = label.cuda()

    pred = net(image)
    pred = torch.nn.functional.softmax(pred, dim=1)

    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            total = total + 1
            results.append((image, torch.max(p.data, 0)[1]))

test_accuracy = total / (itr + 1)
print('Test accuracy {:.8f}'.format(test_accuracy))


   
#results = list()
#total = 0
#for itr, (image, label) in enumerate(test_dataloader):
 
#    if (torch.cuda.is_available()):
#        image = image.cuda()
#        label = label.cuda()
 
#    pred = model(image)
#    pred = torch.nn.functional.softmax(pred, dim=1)
 
#    for i, p in enumerate(pred):
#        if label[i] == torch.max(p.data, 0)[1]:
#            total = total + 1
#            results.append((image, torch.max(p.data, 0)[1]))
 
#test_accuracy = total / (itr + 1)
#print('Test accuracy {:.8f}'.format(test_accuracy))


#pred = net(test_data.data / 255.)
#is_correct = (torch.argmax(pred, dim=1) == test_data.targets
#).float()
#is_correct_percentage = 100 * is_correct.mean()
#print(f'Test accuracy: {is_correct.mean():.4f}')
#print(f'Test percentage accurate: {is_correct_percentage:.2f}'+ "%")

torch.save(net, 'MNIST_classifier_save.pt')
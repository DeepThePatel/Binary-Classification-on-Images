# Imports 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Data Paths
train_data_dir = 'C:/Users/deepk/OneDrive/Documents/College/6th Year/CSCE 580/CSCE580-Fall2023-DeepPatel-Repo/data/pepsico_dataset/Train/'
test_data_dir = 'C:/Users/deepk/OneDrive/Documents/College/6th Year/CSCE 580/CSCE580-Fall2023-DeepPatel-Repo/data/pepsico_dataset/Test/'

batch_size = 32
image_size = (250, 250)

# Transformations (Preprocessing images)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Creating train/test datasets and loaders
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Convolutional neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
model = Net()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Checkpoint filepath
checkpoint_filepath = r'C:/Users/deepk/OneDrive/Documents/College/6th Year/CSCE 580/CSCE580-Fall2023-DeepPatel-Repo/code/pytorch_checkpoint.pt'

# Setup checkpoint
checkpoint_model = Net()
checkpoint = torch.load(checkpoint_filepath)
checkpoint_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
num_epochs = checkpoint['epoch']

# Model evaluation variables
checkpoint_model.eval()
total_val = 0
correct_val = 0

# Loading model checkpoint (includes evaluation and printing)
if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_filepath)
else:
    # Load on CPU
    checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))

# Create model and load state dict
checkpoint_model = Net()
checkpoint_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
num_epochs_checkpoint = checkpoint['epoch']

# Model evaluation on the test dataset
checkpoint_model.eval()
total_val = 0
correct_val = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = checkpoint_model(inputs)
        predicted = torch.round(outputs)
        total_val += labels.size(0)
        correct_val += (predicted == labels.float().view(-1, 1)).sum().item()

# Print accuracy
accuracy = correct_val / total_val
print(f"Accuracy: {accuracy}")
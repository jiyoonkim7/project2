import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Load the data using numpy
data = np.loadtxt("Lidar.samples", dtype=np.double)

# Saving the read out data in numpy arrays
dataY = data[:, -1:] 
dataX = data[:,:-1]

# Transforming numpy arrays to tensors
dataX_tensor = torch.tensor(dataX)
dataY_tensor = torch.tensor(dataY)

# Combine input and output tensors into a tensordataset
dataset = TensorDataset(dataX_tensor, dataY_tensor)

# Define the size of your training and testing subsets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Manual seed for reproducible results
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], torch.Generator().manual_seed(42))

# Use dataloader for batching and iterating
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define the neural network model
from torch import nn
class NeuralNetwork(nn.Module): #from torch import nn
    def __init__(self):
        super().__init__()
        self.linear_1hiddenlayer = nn.Sequential(
            nn.Linear(16, 64), #16 inputs to 1 output
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32,1)
        ) #Loss: 25.9679968911: best result 20/04/2023

    def forward(self, x):
        logits = self.linear_1hiddenlayer(x)
        return logits

# Load neural network
model = NeuralNetwork().double() #keep it float64

# Choosing loss function and optimizer
loss_fn = nn.MSELoss()
#loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

# Train the model 
epochs = 100
epochresults = []
for epoch in range(epochs):
  running_loss = 0.0
  for i, (inputs, labels) in enumerate(train_dataloader):

    outputs = model(inputs)  # forward propagation
    loss = loss_fn(outputs, labels)    
    # to remove previous epoch gradients
    optimizer.zero_grad()    # set optimizer to zero grad
    loss.backward()    
    optimizer.step()
    running_loss += loss.item()  # display statistics
  if not ((epoch + 1) % (epochs // 10)):
    print(f'Epochs:{epoch + 1:5d} | ' \
          f'Batches per epoch: {i + 1:3d} | ' \
          f'Loss: {running_loss / (i + 1):.10f}')
    epochresults.append(running_loss/(i+1))


# save the trained model
PATH = 'simplemodel.pth'
torch.save(model.state_dict(), PATH)
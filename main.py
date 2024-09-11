#A neural network is comprised of layers and modules that perform actions on data. 
#import statement
import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Check the computer's hardware to see if they have a CUDA or GPU to run. If not it defaults to the CPU
#Device setup
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self): #Constructor method to initialize the networks layers and components
        super().__init__() #Calls constructor from the parent class nn.Module
        self.flatten = nn.Flatten() #Reshapes the input tensor. 
        self.linear_relu_stack = nn.Sequential( #Sequence of layers
            nn.Linear(28*28, 512),
            nn.ReLU(), #ReLU activation function that introduces non-linearity
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x): #Defines a forward pass of network. (This is how data flows through network)
        x = self.flatten(x) #Flattens input data
        logits = self.linear_relu_stack(x) #Passes flattened data through the sequence of linear and activation layers
        return logits #Returns the output of netowork

  
model = NeuralNetwork().to(device) #Creates instance of neural network and moves it to device
print(model) #Prints model architectire showing layers and configurations

#Forward pass with dummy data
X = torch.rand(1, 28, 28, device=device) #Creates a random tensor with size (1, 28, 28)
logits = model(X) #Performs a forward pass with the dummy data
pred_probab = nn.Softmax(dim=1)(logits) #Applies the softmax function to convert logits (raw model outputs) into probabilities.
y_pred = pred_probab.argmax(1) #Finds index of hightest probability
print(f"Predicted class: {y_pred}") #Prints the predicted class label

input_image = torch.rand(3,28,28) #Creates a random tensor representing 3 images each size 28x28
print(input_image.size()) #Print size

flatten = nn.Flatten() #Defins flattening layer
flat_image = flatten(input_image) #flattens image into vectors size 784 (28x28)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20) #Defines linear layer with 784 inputs and 20 outputs
hidden1 = layer1(flat_image) #Passes flattened images through linear layer before ReLU activation
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n") #Prints output of linear layer before ReLU
hidden1 = nn.ReLU()(hidden1) #Applies ReLU
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential( 
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
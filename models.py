## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        ## op = (W-F)/S + 1 = (224-5)/1 + 1 = 220
        ## W=Width , F=Filter/Kernel Size , S=Stride
        
#         self.batch1 = nn.BatchNorm2d(32)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ## op = (I-P)/S + 1 = (220-2)/2 + 1 = 110
        ## I=Input , P=Pooling/Kernel Size , S=Stride
        
#         self.dropout1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
#         ## op = (W-F)/S + 1 = (110-3)/1 + 1 = 108
#         ## W=Width , F=Filter/Kernel Size , S=Stride
        
#         self.batch2 = nn.BatchNorm2d(64)
        
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         ## op = (I-P)/S + 1 = (108-2)/2 + 1 = 54
#         ## I=Input , P=Pooling/Kernel Size , S=Stride
        
#         self.dropout2 = nn.Dropout(p=0.2)
        
#         
        
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.fc1 = nn.Linear(54*54*64, 1000)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1000,512)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512, 136)
        
        
        ## Simple model
#         self.conv1 = nn.Conv2d(1, 32, 5)
#         self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.fc = nn.Linear(110*110*32, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv1(x)
        x = self.pool1( F.relu(x) )
        x = self.conv2(x)
        x = self.pool2( F.relu(x) )
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)


        ## Simple
#         x = self.pool( F.relu(self.conv1(x)) )
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

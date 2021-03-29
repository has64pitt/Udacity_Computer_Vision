## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        # out = 1, 112, 112
        
        self.conv1 = nn.Conv2d(1, 64, 3)
        # out = 64, 110, 110
        
        # pool again
        # out = 64, 55, 55

        # pool again
        # out = 64, 27, 27
        
        self.fc1 = nn.Linear(64*27*27, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        #CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        
        x = self.pool(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
        
        
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        dropout_p = 0.01
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 64, 3)
        # output size = (W-F)/S +1 = (224-3)/1 + 1 = 222
        self.norm1 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(p = dropout_p)
        # output size = 64, 222, 222


#         self.conv2 = nn.Conv2d(64, 64, 3)
#         # output size = (W-F)/S +1 = (222-3)/1 + 1 = 220
#         self.norm2 = nn.BatchNorm2d(64)
#         self.drop2 = nn.Dropout2d(p = dropout_p)
#         # output size = 64, 220, 220
        
        ## apply maxpool
        ## output size = 64, 111, 111
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # output size = (W-F)/S +1 = (111-3)/1 + 1 = 109
        self.norm3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p = dropout_p)
        # output size = 128, 109, 109 
        
#         self.conv4 = nn.Conv2d(128, 128, 3)
#         # output size = (W-F)/S +1 = (108-3)/1 + 1 = 106
#         self.norm4 = nn.BatchNorm2d(128)
#         self.drop4 = nn.Dropout2d(p = dropout_p)
#         # output size = 128, 106, 106         
        
        ## apply maxpool
        ## output size = 128, 54, 54
        
        self.conv5 = nn.Conv2d(128, 256, 3)
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        self.norm5 = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(p = dropout_p)
        # output size = 256, 51, 51 
        
#         self.conv6 = nn.Conv2d(256, 256, 3)
#         # output size = (W-F)/S +1 = (51-3)/1 + 1 = 49
#         self.norm6 = nn.BatchNorm2d(256)
#         self.drop6 = nn.Dropout2d(p = dropout_p)
#         # output size = 256, 49, 49         
        
        ## apply maxpool
        ## output size = 256, 24, 24        
        
#         #Linear Layer
        self.fc1 = nn.Linear(256 * 26 * 26, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 136)

        self.pool = nn.MaxPool2d(2, 2)        
            
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        #CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        
        x = self.drop1(F.relu(self.norm1(self.conv1(x))))
        #x = self.drop2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(x)
        
        x = self.drop3(F.relu(self.norm3(self.conv3(x))))
        #x = self.drop4(F.relu(self.norm4(self.conv4(x))))
        x = self.pool(x)
        
        x = self.drop5(F.relu(self.norm5(self.conv5(x))))
        #x = self.drop6(F.relu(self.norm6(self.conv6(x))))
        x = self.pool(x)
       
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)      
        # a modified x, having gone through all the layers of your model, should be returned
        return x

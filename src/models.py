import torch.nn as nn

class FlattenModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FlattenModel, self).__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.layers.forward(X)
    

class ConvModel(nn.Module):
    def __init__(self, conv_layers, linear_layers):
        super(ConvModel, self).__init__()
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers

    def forward(self, X):
        X = self.conv_layers.forward(X)
        X = X.view(X.size(0), -1)
        return self.linear_layers.forward(X)
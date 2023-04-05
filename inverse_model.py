from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['inverse_model']


class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class inverse_model(nn.Module): #ask yihao if cnn?
    def __init__(self, spec_dim=500, layer_dim=6, hidden_dims=None):
        super(inverse_model, self).__init__()
        self.layer_dim = layer_dim
        self.spec_dim = spec_dim
        self.hidden = hidden_dims
        in_channel = spec_dim

        modules = []
        # use default hidden layers if not specified
        if hidden_dims is None:
            #hidden_dims = [256, 128, 64]
            hidden_dims = [128, 64]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    Dense_Block(in_channel, h_dim),
                )
            )
            in_channel = h_dim
        self.fc1 = nn.Sequential(*modules) #500-256-128-64
        self.fc2 = nn.Linear(in_channel, layer_dim) #64-6
        self.relu = torch.nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.abs(x)
        #x = self.sig(x)
        return x

    def loss_function(self, pred, target): #ask yihao
        device = pred.device.type
        wl = torch.linspace(400, 800, self.spec_dim)
        #idx100 = torch.logical_and(wl > 400, wl <= 500)
        #idx80 = torch.logical_and(wl > 500, wl <= 600)
        #idx50 = torch.logical_and(wl > 600, wl <= 700)
        #idx15 = torch.logical_or(wl <= 400, wl > 700)
        idx100 = torch.logical_and(wl >= 530, wl <= 535)
        idx80 = torch.logical_and(wl > 535, wl <= 600)
        idx50 = torch.logical_and(wl > 600, wl <= 700)
        idx15 = torch.logical_or(wl < 535, wl > 500)
        idx40 = torch.logical_and(wl >= 400, wl <= 500)
        weight = torch.ones(1, self.spec_dim)
        weight[0, idx80] = 0.7
        weight[0, idx50] = 0.4
        weight[0, idx15] = 0.7
        weight[0, idx40] = 0.4
        weight = weight.to(device)
        mse = torch.mean(weight * (pred - target) ** 2)
        # loss = torch.mean(mse)
        return mse


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bs = 1024
    spec_dim = 500
    layer_dim = 6
    net = inverse_model(spec_dim=spec_dim, layer_dim=layer_dim).to(device)
    summary(net,(spec_dim,))
    spec = torch.rand(bs, spec_dim).to(device)
    print('predicting...')
    layer = net(spec)
    print('layer.shape = {}'.format(layer.shape))
    print('spec.shape = {}'.format(spec.shape))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from time import time


class conv_norm_relu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', stride=1,
                  bias=True, dilation=1):
        super(conv_norm_relu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size,
                      stride=stride, bias=bias, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self,inputs):
        outputs = self.conv(inputs)
        return outputs
    
    
class CloudSegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.enc_conv0 = conv_norm_relu(3, 16)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv1 = conv_norm_relu(16, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv2 = conv_norm_relu(8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # bottleneck
        self.bottleneck_conv = conv_norm_relu(8, 8)

        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(2, 2)
        self.dec_conv0 = conv_norm_relu(8, 8)

        self.upsample1 = nn.MaxUnpool2d(2, 2)
        self.dec_conv1 = conv_norm_relu(8, 8)

        self.upsample2 = nn.MaxUnpool2d(2, 2)
        self.dec_conv2 = conv_norm_relu(8, 16)

        self.dec3 = conv_norm_relu(16, 4, 5)

    def forward(self, x):
        # encoder
        e0, ind0  = self.pool0(self.enc_conv0(x))
        e1, ind1 = self.pool1(self.enc_conv1(e0))
        e2, ind2 = self.pool2(self.enc_conv2(e1))

        print(e0.shape, e1.shape, ind2.shape)

        # bottleneck
        b = self.bottleneck_conv(e2)

        print(b.shape, ind2.shape)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b, ind2))
        d1 = self.dec_conv1(self.upsample1(d0, ind1))
        d2 = self.dec_conv2(self.upsample2(d1, ind0))

        d3 = self.dec3(d2)

        return d3
    

def train(model, opt, loss_fn, epochs, data_tr, device):
    train_trace, val_trace = [], []
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)
            loss.backward()
            opt.step()

            # calculate loss to show the user
            avg_loss += loss.detach().cpu().numpy() / len(data_tr)
        train_trace.append(avg_loss)
        toc = time()
        print('loss: %f' % avg_loss)

    return
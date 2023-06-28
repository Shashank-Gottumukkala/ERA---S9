import torch.nn as nn
import torchinfo

class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, bias = False, padding = 0, dws= False, skip = False, dilation = 1, dropout = 0):
        super(ConvLayer, self).__init__

        self.skip = skip

        if dws and input_channels == output_channels:
            self.convlayer == nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, bias=bias, padding=padding, groups = input_channels, dilation= dilation, padding_mode= 'replicate'),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 1, bias=bias)
            )

        else:
            self.convlayer = nn.Conv2d(input_channels, output_channels, 3, bias= bias, padding=padding, groups=1, dilation=dilation, padding_mode='replicate'),
            self.normlayer = nn.BatchNorm2d(output_channels)

        self.skiplayer = None
        if self.skip and input_channels != output_channels:
            self.skiplayer = nn.Conv2d(input_channels, output_channels, 1, bias=bias)

        self.activation = nn.ReLU()

        self.dropout = None
        if dropout > 0:
            self.droplayer = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.convlayer(x)
        x = self.normlayer(x)
        if self.skip:
            if self.skiplayer is None:
                x += x_
            else:
                x += self.skiplayer(x_)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


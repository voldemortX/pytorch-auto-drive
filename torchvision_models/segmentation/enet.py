# Enet pytorch code retrieved from https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py
import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    """
    1. A conv branch performs a regular convolution with stride 2;
    2. An extension branch performs max-pooling.
    Doing both operations in parallel and concatenating their results allows for efficient downsampling and expansion.
    The conv branch outputs 13 feature maps while the maxpool branch outputs 3, for a total of 16 feature maps after concatenation.
    """

    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()

        if relu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.PReLU()

        self.conv_branch = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=3, padding=1, bias=bias)
        self.maxpool_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        conv = self.conv_branch(x)
        maxpool = self.maxpool_branch(x)
        out = torch.cat((conv, maxpool), dim=1)
        out = self.batch_norm(out)

        return self.activation(out)


class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0, dilation=1, asymmetric=False, dropout=0,
                 bias=False, relu=False):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                "Value out of range. Expected value in the interval [1,{0}], got internal_scale={1}".format(channels,
                                                                                                            internal_ratio))
        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a  regularize (spatial dropout). Number of channel is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        # if the convolution is asymmetric we split the main convolution in
        # two. e.g., for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1,
                          padding=(padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=1,
                          padding=(0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation()
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding,
                          dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation()
            )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation()
        )

        self.ext_regul = nn.Dropout2d(p=dropout)
        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        # Add main and extension branches
        out = main + ext
        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, return_indices=False, dropout=0, bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        # convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation()
        )
        self.ext_regul = nn.Dropout2d(p=dropout)

        # PReLU layer to apply after concatenationg the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.reture_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before cancatenating, check if main is on the CPU or GPU and convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # add main and extension branches
        out = main + ext
        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout=0, bias=False, relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        # Remember that the stride is the same as the kernel_size, just like the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2=nn.Sequential(
            nn.Conv2d(internal_channels,out_channels,kernel_size=1,bias=bias),
            nn.BatchNorm2d(out_channels),
            activation()
        )
        self.ext_regul=nn.Dropout2d(p=dropout)
        self.out_activation=activation()

    def forward(self,x,max_indices,output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)



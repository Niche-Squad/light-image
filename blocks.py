"""
modified from the repository alexandru-dinu/cae:
https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_32x32x32_zero_pad_bin.py
"""

import torch.nn as nn

def ConvBlockA(cin, cout, k, s):
    return nn.Sequential(
        nn.ZeroPad2d((1, 2, 1, 2)),
        nn.Conv2d(
            in_channels=cin, 
            out_channels=cout, 
            kernel_size=(k, k), 
            stride=(s, s)
        ),
        nn.LeakyReLU(),
    )

def ConvBlockB(cin, cout, k, s):
    return nn.Sequential(
        nn.ZeroPad2d((1, 1, 1, 1)),
        nn.Conv2d(
            in_channels=cin, 
            out_channels=cout, 
            kernel_size=(k, k), 
            stride=(s, s)
        ),
        nn.LeakyReLU(),
        nn.ZeroPad2d((1, 1, 1, 1)),
        nn.Conv2d(
            in_channels=cout, 
            out_channels=cout, 
            kernel_size=(k, k), 
            stride=(s, s)
        ),
    )

def ConvBlockC(cin, cout, k, s, p):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p),
        ),
        nn.Tanh(),
    )

def ConvBlockC16(cin, cout, k, s):
    return nn.Sequential(
        nn.ZeroPad2d((1, 2, 1, 2)),
        nn.Conv2d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=(k, k),
            stride=(s, s),
        ),
        nn.Tanh(),
    )
    
    
def ConvBlockD(cin, cout, k, s):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=cin, 
            out_channels=cout, 
            kernel_size=(k, k), 
            stride=(s, s)
        ),
        nn.LeakyReLU(),
        nn.ReflectionPad2d((2, 2, 2, 2)),
        nn.Conv2d(
            in_channels=cout, 
            out_channels=3, 
            kernel_size=(k, k), 
            stride=(s, s)
        ),
        nn.Tanh(),
    )


def ConvBlockD16(cin, cout, k, s):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=cin, 
            out_channels=cout, 
            kernel_size=(k, k), 
            stride=(s, s)
        ),
        nn.LeakyReLU(),
        nn.ZeroPad2d((1, 1, 1, 1)),
        nn.ConvTranspose2d(
            in_channels=cout, 
            out_channels=3, 
            kernel_size=(2, 2), 
            stride=(2, 2)
        ),
        nn.Tanh(),
    )

def TransConvBlock(cin, cmid, cout):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=cin, 
            out_channels=cmid, 
            kernel_size=(3, 3), 
            stride=(1, 1)
        ),
        nn.LeakyReLU(),
        nn.ZeroPad2d((1, 1, 1, 1)),
        nn.ConvTranspose2d(
            in_channels=cmid, 
            out_channels=cout, 
            kernel_size=(2, 2), 
            stride=(2, 2)
        ),
    )
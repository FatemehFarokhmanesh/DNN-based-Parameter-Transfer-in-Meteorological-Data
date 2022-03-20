import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddingBlock2d(nn.Module):
    def __init__(self, padding_size, padding_mode=None):
        super(PaddingBlock2d, self).__init__()
        self.padding_size = self._parse_padding_size(padding_size)
        self.padding_mode = self._parse_padding_mode(padding_mode)

    def forward(self, x):
        vpad = self.padding_size[:2] + [0, 0]
        hpad = [0, 0] + self.padding_size[2:]
        return F.pad(
            F.pad(x, hpad, mode=self.padding_mode[1]),
            vpad, mode=self.padding_mode[0]
        )

    @staticmethod
    def _parse_padding_size(padding_size):
        if isinstance(padding_size, int):
            out = [padding_size,] * 4
        elif isinstance(padding_size, (tuple, list)):
            if len(padding_size) == 2:
                out = ([padding_size[0]] * 2) + ([padding_size[1]] * 2)
            elif len(padding_size) == 4:
                out = list(padding_size)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return out

    @staticmethod
    def _parse_padding_mode(padding_mode):
        if padding_mode is None:
            padding_mode = ('reflect', 'circular')
        elif isinstance(padding_mode, tuple):
            assert len(padding_mode) == 2
            for p in padding_mode:
                assert p in ['constant', 'reflect', 'replicate', 'circular']
            padding_mode = padding_mode
        elif isinstance(padding_mode, str):
            assert padding_mode in ['constant', 'reflect', 'replicate', 'circular']
            padding_mode = (padding_mode, padding_mode)
        else:
            raise NotImplementedError()
        return padding_mode
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, padding_mode=None):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels

        self.in_layer_encoder = nn.Sequential(       #64,16,32
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.in_channels, self.inner_channels, kernel_size=3, stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU()
        )
        self.hr_conv_layer_encoder = nn.Sequential(      #128,8,16
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels, self.inner_channels * 2, kernel_size=3,  stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU()
        )
        self.mr_conv_layer_encoder = nn.Sequential(     #256,4,8
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels* 2, self.inner_channels * 4, kernel_size=3, stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU()
        )
        self.lr_conv_layer_encoder = nn.Sequential(     #512,1,1
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels * 4, self.inner_channels * 8, kernel_size=3, stride=(4, 8)),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU()
        )

        self.in_layer_decoder = nn.Sequential(     #256,4,8
            nn.ConvTranspose2d(self.inner_channels * 8, self.inner_channels * 4, kernel_size=(4, 8)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels * 4, self.inner_channels * 4, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU()
        )

        self.lr_conv_layer_decoder = nn.Sequential(     #128,8,16
            nn.ConvTranspose2d(self.inner_channels * 8, self.inner_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels * 2, self.inner_channels * 2, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU()
        )

        self.mr_conv_layer_decoder = nn.Sequential(     #64,16,32
            nn.ConvTranspose2d(self.inner_channels * 4, self.inner_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU()
        )

        self.hr_conv_layer_decoder = nn.Sequential(     #1,32,64
            nn.ConvTranspose2d(self.inner_channels * 2, self.out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3),
            nn.BatchNorm2d(self.out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        #encoder
        features1 = self.in_layer_encoder(x)
        features2 = self.hr_conv_layer_encoder(features1)
        features3 = self.mr_conv_layer_encoder(features2)
        features4 = self.lr_conv_layer_encoder(features3)
        features5 = self.in_layer_decoder(features4)
        features6 = torch.cat([features3, features5], dim=1)
        features7 = self.lr_conv_layer_decoder(features6)
        features8 = torch.cat([features2, features7], dim=1)
        features9 = self.mr_conv_layer_decoder(features8)
        features10 = torch.cat([features1, features9], dim=1)
        features11 = self.hr_conv_layer_decoder(features10)
        return features11

if __name__ == '__main__':
    unet = UNet(1, 1, 64)
    print(unet)
    a = torch.randn(10, 1, 32, 64)
    c = unet(a)
    print(c.shape)

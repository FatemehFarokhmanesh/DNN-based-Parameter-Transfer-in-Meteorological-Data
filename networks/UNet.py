import argparse
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


class SkipConnection(nn.Module):

    def __init__(self, inner_module: nn.Module):
        super(SkipConnection, self).__init__()
        self.inner_module = inner_module

    def forward(self, x):
        return torch.cat([x, self.inner_module(x)], dim=1)


class UNet(nn.Module):

    @staticmethod
    def init_parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        group = parser.add_argument_group('unet')
        group.add_argument('--model:unet:hidden-channels', type=int, default=None,
                           help='hidden channels for unet model')
        group.add_argument('--model:unet:depth', type=int, default=3,
                           help='number of skip connections in unet model')
        group.add_argument('--model:unet:padding-mode', type=str, default='replicate',
                           help='padding mode for unet model')
        group.add_argument('--model:unet:activation', type=str, default='LeakyReLU',
                           help='nonlinear activation for unet model')
        group.add_argument('--model:unet:dropout', type=float, default=0., help='dropout rate for unet model')
        return parser

    @classmethod
    def from_dict(cls, in_channels, out_channels, args):
        hidden_channels = args['model:unet:hidden_channels']
        assert hidden_channels is not None, '[ERROR] Use of UNet model requires setting number of hidden channels!'
        depth = args['model:unet:depth']
        padding_mode = args['model:unet:padding_mode']
        return cls(
            in_channels, hidden_channels, out_channels, depth=depth, padding_mode=padding_mode,
            activation=args['model:unet:activation'], dropout=args['model:unet:dropout']
        )

    def __init__(self, in_channels, hidden_channels, out_channels, depth=3, padding_mode='replicate', activation='LeakyReLU', dropout=0.):
        super(UNet, self).__init__()
        activation_class = getattr(nn, activation)
        in_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (5, 5), padding_mode=padding_mode, padding=(2, 2)),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout2d(dropout),
            activation_class()
        )
        upper_channels = hidden_channels * 2 ** (depth - 1)
        lower_channels = hidden_channels * 2 ** depth
        unet = nn.Sequential(
            nn.Conv2d(upper_channels, lower_channels, (3, 3), padding=(1, 1), padding_mode=padding_mode, stride=(2, 2)),
            nn.BatchNorm2d(lower_channels),
            activation_class(),
            nn.Conv2d(lower_channels, lower_channels, (1, 1)),
            nn.BatchNorm2d(lower_channels),
            activation_class(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(lower_channels, upper_channels, (4, 4), padding_mode='zeros', padding=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(upper_channels),
            activation_class(),
        )
        for i in range(1, depth):
            upper_channels = hidden_channels * 2 ** (depth - i - 1)
            lower_channels = hidden_channels * 2 ** (depth - i)
            unet = nn.Sequential(
                nn.Conv2d(upper_channels, lower_channels, (3, 3), padding_mode=padding_mode, padding=(1, 1), stride=(2, 2)),
                nn.BatchNorm2d(lower_channels),
                nn.Dropout2d(dropout),
                activation_class(),
                SkipConnection(unet),
                nn.Conv2d(2 * lower_channels, lower_channels, (3, 3), padding=(1, 1), padding_mode=padding_mode),
                nn.BatchNorm2d(lower_channels),
                nn.Dropout2d(dropout),
                activation_class(),
                nn.ConvTranspose2d(lower_channels, upper_channels, (4, 4), padding_mode='zeros', padding=(1, 1), stride=(2, 2)),
                nn.BatchNorm2d(upper_channels),
                activation_class(),
            )
        assert upper_channels == hidden_channels, f'[ERROR] upper channels: {upper_channels}, hidden_channels: {hidden_channels}'
        out_layer = nn.Sequential(
            nn.Conv2d(2 * upper_channels, upper_channels, (3, 3), padding=(1, 1), padding_mode=padding_mode),
            nn.BatchNorm2d(upper_channels),
            activation_class(),
            nn.Conv2d(upper_channels, out_channels, (3, 3), padding=(1, 1), padding_mode=padding_mode)
        )
        self.model = nn.Sequential(
            in_layer,
            SkipConnection(unet),
            out_layer,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    unet = UNet(1, 1, 64)
    print(unet)
    a = torch.randn(10, 1, 32, 64)
    c = unet(a)
    print(c.shape)

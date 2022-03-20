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


class ResBlock(nn.Module):
    def __init__(self, num_channels, num_layers, padding_mode=None, use_oa=False):
        super(ResBlock, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.block = None
        self.output_activation = None
        self._build_block(padding_mode, use_oa)

    def _build_block(self, padding_mode, use_oa):
        layers = []
        for i in range(self.num_layers - 1):
            layers += [
                PaddingBlock2d((1, 1), padding_mode),
                nn.Conv2d(self.num_channels, self.num_channels, 3),
                nn.BatchNorm2d(self.num_channels),
                nn.LeakyReLU(),
            ]
        layers += [
            PaddingBlock2d((1, 1), padding_mode),
            nn.Conv2d(self.num_channels, self.num_channels, 3),
            nn.BatchNorm2d(self.num_channels),
        ]
        self.block = nn.Sequential(*layers)
        if use_oa:
            self.output_activation = nn.LeakyReLU()

    def forward(self, x):
        out = x + self.block(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

class SimpleResnet(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, num_layers, padding_mode=None):
        super(SimpleResnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers
        self.in_layer = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.in_channels, self.latent_channels, 3),
            nn.BatchNorm2d(self.latent_channels),
            nn.LeakyReLU()
        )
        self.res_block = ResBlock(self.latent_channels, self.num_layers, padding_mode=padding_mode, use_oa=True)

        self.out_layer = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.latent_channels, out_channels, 3),
            
        )

    

    def forward(self, x):
        features = self.in_layer(x)
        features = self.res_block(features)
        features = self.res_block(features)
        features = self.res_block(features)
        return self.out_layer(features)


if __name__ == '__main__':
    residual_block = ResBlock(4, 2)
    residual_net = SimpleResnet(4, 4, 64, 2, 2)
    a = torch.randn(10, 4, 32, 64)
    b = residual_block(a)
    c = residual_net(a)
    print(b.shape, c.shape)

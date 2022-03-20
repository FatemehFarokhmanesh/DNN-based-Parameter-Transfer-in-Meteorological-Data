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

class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_layers, padding_mode=None, use_oa=False):
        super(ConvBlock, self).__init__()
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
                #nn.Dropout(0.2),
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
        out = self.block(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, num_channels, num_layers, use_oa=False):
        super(DeconvBlock, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.block = None
        self.output_activation = None
        self._build_block(use_oa)

    def _build_block(self, use_oa):
        layers = []
        for i in range(self.num_layers - 1):
            layers += [
                nn.ConvTranspose2d(self.num_channels, self.num_channels, 3, padding=1),
                nn.BatchNorm2d(self.num_channels),
                nn.LeakyReLU(),
            ]
        layers += [
            nn.ConvTranspose2d(self.num_channels, self.num_channels, 3, padding=1),
            nn.BatchNorm2d(self.num_channels),
        ]
        self.block = nn.Sequential(*layers)
        if use_oa:
            self.output_activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.block(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out
        
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, bottleneck_channels, num_layers, padding_mode=None):
        super(ConvAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.bottleneck_channels = bottleneck_channels
        self.num_layers = num_layers
        self.in_layer_encoder = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.in_channels, self.latent_channels, 3),
            nn.BatchNorm2d(self.latent_channels),
            nn.LeakyReLU()
        )
        self.conv_block = ConvBlock(self.latent_channels, self.num_layers, padding_mode=padding_mode, use_oa=True)

        self.out_layer_encoder = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.latent_channels, self.bottleneck_channels, 3),
        )

        self.in_layer_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.bottleneck_channels, self.latent_channels, 3, padding=1),
            nn.BatchNorm2d(self.latent_channels),
            nn.LeakyReLU()
        )
        self.deconv_block = DeconvBlock(self.latent_channels, self.num_layers, use_oa=True)

        self.out_layer_decoder = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.latent_channels, self.out_channels, 3),
        )
        

    def forward(self, x):
        #encoder
        features = self.in_layer_encoder(x)
        features = self.conv_block(features)
        features = self.out_layer_encoder(features)
        #decoder
        features = self.in_layer_decoder(features)
        features = self.deconv_block(features)
        return self.out_layer_decoder(features)

if __name__ == '__main__':
    ae = ConvAutoencoder(7, 6, 112, 16, 2)
    print(ae)
    a = torch.randn(10, 7, 32, 64)
    c = ae(a)
    print(c.shape)

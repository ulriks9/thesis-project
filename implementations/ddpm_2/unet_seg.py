from collections import OrderedDict

import torch
import torch.nn as nn


class UNet_Seg(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, timesteps=1000):
        super(UNet_Seg, self).__init__()
        self.device = torch.device('cuda:0')
        # Time embedding:
        self.timesteps = timesteps
        self.time_embed = nn.Embedding(self.timesteps, self.timesteps)
        self.time_embed.weight.data = self.sinusoidal_embedding(self.timesteps, self.timesteps)
        self.time_embed.requires_grad_(False)
        self.in_channels = in_channels
        self.out_channels = out_channels

        features = init_features
        self.time_embedding1 = self._make_te(self.timesteps, self.in_channels)
        self.encoder1 = UNet_Seg._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_embedding2 = self._make_te(self.timesteps, features)
        self.encoder2 = UNet_Seg._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_embedding3 = self._make_te(self.timesteps, features * 2)
        self.encoder3 = UNet_Seg._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_embedding4 = self._make_te(self.timesteps, features * 4)
        self.encoder4 = UNet_Seg._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.time_embedding_bottleneck = self._make_te(self.timesteps, features * 8)
        self.bottleneck = UNet_Seg._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.time_embedding5 = self._make_te(self.timesteps, features * 16)
        self.decoder4 = UNet_Seg._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.time_embedding6 = self._make_te(self.timesteps, features * 8)
        self.decoder3 = UNet_Seg._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.time_embedding7 = self._make_te(self.timesteps, features * 4)
        self.decoder2 = UNet_Seg._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.time_embedding8 = self._make_te(self.timesteps, features * 2)
        self.decoder1 = UNet_Seg._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, t):
        n = len(x)
        x = x.to(self.device)
        t = self.time_embed(t)

        enc1 = self.encoder1(x + self.time_embedding1(t).reshape(n, -1, 1, 1))
        enc2 = self.encoder2(self.pool1(enc1) + self.time_embedding2(t).reshape(n, -1, 1, 1))
        enc3 = self.encoder3(self.pool2(enc2) + self.time_embedding3(t).reshape(n, -1, 1, 1))
        enc4 = self.encoder4(self.pool3(enc3) + self.time_embedding4(t).reshape(n, -1, 1, 1))

        bottleneck = self.bottleneck(self.pool4(enc4) + self.time_embedding_bottleneck(t).reshape(n, -1, 1, 1))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4 + self.time_embedding5(t).reshape(n, -1, 1, 1))
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3 + self.time_embedding6(t).reshape(n, -1, 1, 1))
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2 + self.time_embedding7(t).reshape(n, -1, 1, 1))
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1 + self.time_embedding8(t).reshape(n, -1, 1, 1))
        
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def sinusoidal_embedding(self, n, d):
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:,::2] = torch.sin(t * wk[:,::2])
        embedding[:,1::2] = torch.cos(t * wk[:,::2])

        return embedding.to(self.device)
    
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
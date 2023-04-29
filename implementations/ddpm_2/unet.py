import torch
import torch.nn as nn
from config import TrainingConfig
from tqdm import tqdm
from utils import print_progress

# UNet implementation with 3 blocks in encoder and decoder
# Monte Carlo Dropout is not yet implemented
class UNet(nn.Module):
    def __init__(
            self, 
            timesteps,
            in_channels=1, 
            out_channels=1,
            features=None,
            kernel_size=3,
            padding=1, 
            stride=1,
            down_pool_kernel_size=2,
            up_conv_kernel_size=2,
            pool_stride=2
        ):
        
        super().__init__()

        # This was recommended by SonarLint:
        if features == None:
            self.features = [64, 128, 256]
        else:
            self.features = features

        self.device = torch.device('cuda:0')
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.down_pool_kernel_size = down_pool_kernel_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.padding = padding
        self.stride = stride
        self.pool_stride = pool_stride
        self.config = TrainingConfig

        # Time embedding:
        self.time_embed = nn.Embedding(self.timesteps, self.timesteps)
        self.time_embed.weight.data = self.sinusoidal_embedding(self.timesteps, self.timesteps)
        self.time_embed.requires_grad_(False)

        # Encoder blocks:
        self.time_embedding1 = self._make_te(self.timesteps, self.in_channels)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.features[0], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[0], out_channels=self.features[0], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=self.down_pool_kernel_size, stride=self.pool_stride)

        self.time_embedding2 = self._make_te(self.timesteps, self.features[0])
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=self.features[0], out_channels=self.features[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[1], out_channels=self.features[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[1]),
            nn.ReLU(inplace=True)
        )
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=self.down_pool_kernel_size, stride=self.pool_stride)
            
        self.time_embedding3 = self._make_te(self.timesteps, self.features[1])
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=self.features[1], out_channels=self.features[2], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[2], out_channels=self.features[2], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[2]),
            nn.ReLU(inplace=True)
        )
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=self.down_pool_kernel_size, stride=self.pool_stride)

        # Bottleneck:
        self.time_embedding_bottleneck = self._make_te(self.timesteps, self.features[2])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.features[2], out_channels=self.features[2] * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[2] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[2] * 2, out_channels=self.features[2] * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[2] * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder blocks:
        self.decoder_up_1 = nn.ConvTranspose2d(in_channels=self.features[2] * 2, out_channels=self.features[2], kernel_size=self.up_conv_kernel_size, stride=self.pool_stride)
        self.time_embedding4 = self._make_te(self.timesteps, self.features[2] * 2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=self.features[2] * 2, out_channels=self.features[2], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[2], out_channels=self.features[2], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[2]),
            nn.ReLU(inplace=True)
        )

        self.decoder_up_2 = nn.ConvTranspose2d(in_channels=self.features[2], out_channels=self.features[1], kernel_size=self.up_conv_kernel_size, stride=self.pool_stride)
        self.time_embedding5 = self._make_te(self.timesteps, self.features[2])
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=self.features[2], out_channels=self.features[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[1], out_channels=self.features[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[1]),
            nn.ReLU(inplace=True)
        )

        self.decoder_up_3 = nn.ConvTranspose2d(in_channels=self.features[1], out_channels=self.features[0], kernel_size=self.up_conv_kernel_size, stride=self.pool_stride)
        self.time_embedding_out = self._make_te(self.timesteps, self.features[1])
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=self.features[1], out_channels=self.features[0], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[0], out_channels=self.features[0], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[0]),
            nn.ReLU(inplace=True)
        )

        # Last convolution layer:
        self.final_conv = nn.Conv2d(in_channels=self.features[0], out_channels=self.out_channels, kernel_size=1)

    def forward(self, x, t, return_dict=False):
        n = len(x)
        x = x.to(self.device)
        t = self.time_embed(t)

        enc1 = self.encoder1(x + self.time_embedding1(t).reshape(n, -1, 1, 1))
        enc2 = self.encoder2(self.encoder_pool1(enc1) + self.time_embedding2(t).reshape(n, -1, 1, 1))
        enc3 = self.encoder3(self.encoder_pool2(enc2) + self.time_embedding3(t).reshape(n, -1, 1, 1))

        bottleneck = self.bottleneck(self.encoder_pool3(enc3) + self.time_embedding_bottleneck(t).reshape(n, -1, 1, 1))

        dec1 = self.decoder_up_1(bottleneck)
        dec1 = torch.cat((dec1, enc3), dim=1)
        dec1 = self.decoder1(dec1 + self.time_embedding4(t).reshape(n, -1, 1, 1))
        dec2 = self.decoder_up_2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2 + self.time_embedding5(t).reshape(n, -1, 1, 1))
        dec3 = self.decoder_up_3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec3 = self.decoder3(dec3 + self.time_embedding_out(t).reshape(n, -1, 1, 1))
        dec3 = self.final_conv(dec3)

        return torch.sigmoid(dec3)
    
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

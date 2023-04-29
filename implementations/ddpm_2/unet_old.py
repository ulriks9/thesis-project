import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None, kernel_size=3, padding=1, stride=2):
        super().__init__()

        # This was recommended by SonarLint:
        if features == None:
            self.features = [64, 128, 256]
        else:
            self.features = features

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        count = 0
        # Create encoder blocks:
        for f in self.features:
            # If first iteration, input channels should be input image:
            if count == 0:
                in_channels = self.in_channels
            else:
                in_channels = prev_f

            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(num_features=f),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=f, out_channels=f, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True)
            )

            count += 1
            prev_f = f
            self.encoder_blocks.append(conv_block)
            self.encoder_blocks.append(nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride))
            
        # Bottleneck:
        bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.features[-1], out_channels=self.features[-1] * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.features[-1] * 2, out_channels=self.features[-1] * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=self.features[-1] * 2),
            nn.ReLU(inplace=True)
        )

        self.encoder_blocks.append(bottleneck)

        # Create decoder blocks:
        for idx, f in enumerate(reversed(self.features)):
            # Get the next feature number from the list:
            if idx != len(self.features) - 1:
                next_f = list(reversed(self.features))[idx + 1]
            else:
                next_f = int(self.features[0] / 2)

            # Transposed convolution:
            up_conv = nn.ConvTranspose2d(in_channels=f * 2, out_channels=next_f * 2, kernel_size=self.kernel_size, stride=self.stride)
            
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=f * 2, out_channels=next_f * 2, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(num_features=next_f * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=next_f * 2, out_channels=next_f * 2, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(num_features=next_f),
                nn.ReLU(inplace=True)
            )

            self.decoder_blocks.append(up_conv)
            self.decoder_blocks.append(conv_block)

        # Last convolution layer:
        self.decoder_blocks.append(nn.Conv2d(in_channels=self.features[0], out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding))

    def forward(self, x):
        encoder_outputs = []

        # Encoder:
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
        
        # Decoder:
        for idx, block in enumerate(self.decoder_blocks):
            print(encoder_outputs[idx].shape)
            x = block(torch.cat((x, encoder_outputs[-idx - 2]), dim=1) + encoder_outputs[-idx - 3])
            

        return x




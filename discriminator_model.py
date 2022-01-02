import torch
import torch.nn as nn
from ciconv2d import CIConv2d


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_ciconv=False, use_cycle_wgan=False):
        super().__init__()
        self.use_ciconv = use_ciconv
        self.use_cycle_wgan = use_cycle_wgan
        if use_ciconv:
            self.ciconv = CIConv2d('W', k=3, scale=0.0)
            in_channels = 1
        self.initial = nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_ciconv:
            x = self.ciconv(x)
        x = self.initial(x)
        x = self.leaky(x)
        x = self.model(x)
        if self.use_cycle_wgan:
            return x
        return torch.sigmoid(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()

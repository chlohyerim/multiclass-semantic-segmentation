from torch import nn


# 3x3 convolutional layer 2번
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.sequence(x)
    

class AttentionGate(nn.Module):
    def __init__(self, in_channels, attention_channels):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=attention_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(attention_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=attention_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, g):
        x1 = self.conv2d(x)
        g1 = self.conv2d(g)

        psi = self.relu(x1 + g1)  # element-wise add
        psi = self.psi(psi)

        return x * psi  # element-wise multiply
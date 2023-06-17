import torch
from torch import nn

from nets import blocks


# 모델 구성
class Net(nn.Module):
    def __init__(self, n_class, c_x1, padding_mode):
        super(Net, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = blocks.DoubleConvBlock(c_in=3, c_out=c_x1, padding_mode=padding_mode)
        self.encoder2 = blocks.DoubleConvBlock(c_in=c_x1, c_out=c_x1 * 2, padding_mode=padding_mode)
        self.encoder3 = blocks.DoubleConvBlock(c_in=c_x1 * 2, c_out=c_x1 * 4, padding_mode=padding_mode)
        self.encoder4 = blocks.DoubleConvBlock(c_in=c_x1 * 4, c_out=c_x1 * 8, padding_mode=padding_mode)
        self.encoder5 = blocks.DoubleConvBlock(c_in=c_x1 * 8, c_out=c_x1 * 16, padding_mode=padding_mode)

        self.decoder5 = blocks.DoubleConvBlock(c_in=c_x1 * 16, c_out=c_x1 * 8, padding_mode=padding_mode)
        self.decoder4 = blocks.DoubleConvBlock(c_in=c_x1 * 8, c_out=c_x1 * 4, padding_mode=padding_mode)
        self.decoder3 = blocks.DoubleConvBlock(c_in=c_x1 * 4, c_out=c_x1 * 2, padding_mode=padding_mode)
        self.decoder2 = blocks.DoubleConvBlock(c_in=c_x1 * 2, c_out=c_x1, padding_mode=padding_mode)

        self.upconv5 = nn.ConvTranspose2d(in_channels=c_x1 * 16, out_channels=c_x1 * 8, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=c_x1 * 8, out_channels=c_x1 * 4, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=c_x1 * 4, out_channels=c_x1 * 2, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=c_x1 * 2, out_channels=c_x1, kernel_size=2, stride=2)

        self.attention5 = blocks.AttentionGate(c_in=c_x1 * 8, c_att=c_x1 * 4)
        self.attention4 = blocks.AttentionGate(c_in=c_x1 * 4, c_att=c_x1 * 2)
        self.attention3 = blocks.AttentionGate(c_in=c_x1 * 2, c_att=c_x1)
        self.attention2 = blocks.AttentionGate(c_in=c_x1, c_att=c_x1 // 2)

        self.fconv = nn.Conv2d(in_channels=c_x1, out_channels=n_class, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight)


    def forward(self, x):
        # encoding path
        x1 = self.encoder1(x)

        x2 = self.maxpool(x1)
        x2 = self.encoder2(x2)

        x3 = self.maxpool(x2)
        x3 = self.encoder3(x3)

        x4 = self.maxpool(x3)
        x4 = self.encoder4(x4)

        x5 = self.maxpool(x4)
        x5 = self.encoder5(x5)

        # decoding path
        d5 = self.upconv5(x5)
        x4 = self.attention5(x=x4, g=d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        x3 = self.attention4(x=x3, g=d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        x2 = self.attention3(x=x2, g=d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        x1 = self.attention2(x=x1, g=d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.decoder2(d2)

        # 1x1 conv
        d1 = self.fconv(d2)

        return d1
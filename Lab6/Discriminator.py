from torch import nn


class FrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super(FrontEnd, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.discriminator(x).view(-1)
        return output


class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()

        # self.Q = nn.Sequential(
        #     nn.Linear(8192, 100, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(100, 10, bias=True)

        self.conv = nn.Conv2d(512, 100, 1, bias=False)
        self.bn = nn.BatchNorm2d(100)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(100, 10, 1)
        self.conv_mu = nn.Conv2d(100, 2, 1)
        self.conv_var = nn.Conv2d(100, 2, 1)

    def forward(self, x):
        y = self.conv(x)
        disc_logits = self.conv_disc(y).squeeze()

        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var

        # s = x.size(0) * x.size(1) * x.size(2) * x.size(3)
        # s = (int)(s / 8192)
        # output = self.Q(x.view(s, 8192)).squeeze()
        # return output
import torch
import torch.nn as nn

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
#                 nn.InstanceNorm2d(nout, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
#                 nn.InstanceNorm2d(nout, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1, conditional=False):
        super(encoder, self).__init__()
        self.conditional = conditional
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        # input condition: global content vector
        if conditional:
            self.c2 = dcgan_conv(nf + 2, nf * 2)
        else:
            self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
#                 nn.InstanceNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input, cond=None):
        if cond is not None:
            assert cond.size(1) % 32 == 0
            W = cond.size(1) // 32
            R = 32 // W
            ch_1 = cond.view(-1, 1, W, 32)
            ch_1 = ch_1.repeat(1, 1, R, 1)
            ch_2 = cond.view(-1, 1, 32, W)
            ch_2 = ch_2.repeat(1, 1, 1, R)
        
        h1 = self.c1(input)
        if self.conditional:
            h2 = self.c2(torch.cat([h1, ch_1, ch_2], 1))
        else:
            h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
#                 nn.InstanceNorm2d(nf * 8, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output


import torch
import torch.nn as nn

class discriminator(nn.Module):
    def __init__(self, pose_dim, nf=512):
        super(discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
                nn.Linear(pose_dim*2, nf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(nf, nf//2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(nf//2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        output = self.main(torch.cat(input, 1).view(-1, self.pose_dim*2))
        return output
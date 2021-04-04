import torch.nn as nn


class CSA(nn.Module):
    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, a):
        r = self.encoder(x)
        r0 = r[a == 0]
        r1 = r[a == 1]
        t = self.decoder(r0=r0, r1=r1)
        return t, r0, r1

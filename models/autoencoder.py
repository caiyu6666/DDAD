import torch.nn as nn


class AE(nn.Module):
    def __init__(self, latent_size, multiplier=1, unc=False, img_size=64, vae=False):
        super(AE, self).__init__()
        out_channels = 2 if unc else 1
        self.fm = img_size // 16
        self.unc = unc
        self.mp = multiplier
        self.encoder = nn.Sequential(
            nn.Conv2d(1, int(16 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(16 * multiplier),
                      int(32 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(32 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
        )
        if not vae:
            self.linear_enc = nn.Sequential(
                nn.Linear(int(64 * multiplier) * self.fm * self.fm, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, latent_size),
            )
        else:
            self.linear_enc = nn.Sequential(
                nn.Linear(int(64 * multiplier) * self.fm * self.fm, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, latent_size * 2),
            )

        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, int(64 * multiplier) * self.fm * self.fm),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(64 * multiplier), int(64 *
                                                         multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(64 * multiplier), int(32 *
                                                         multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(32 * multiplier), int(16 *
                                                         multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(16 * multiplier),
                               out_channels, 4, 2, 1, bias=False),
        )
        # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        lat_rep = self.feature(x)
        out = self.decode(lat_rep)
        return out

    def feature(self, x):
        lat_rep = self.encoder(x)
        lat_rep = lat_rep.view(lat_rep.size(0), -1)
        lat_rep = self.linear_enc(lat_rep)
        return lat_rep

    def decode(self, x):
        out = self.linear_dec(x)
        out = out.view(out.size(0), int(64 * self.mp), self.fm, self.fm)
        out = self.decoder(out)
        if not self.unc:
            return out
        else:
            return out.chunk(2, 1)

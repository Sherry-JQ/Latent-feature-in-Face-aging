import torch
import torch.nn as nn
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class CVAE(nn.Module):
    def __init__(self, img_dim=3, z_dim=1024, layer_size=4, c_dim=3):
        super(CVAE, self).__init__()

        self.layer_size = layer_size
        # Encoder
        cin = img_dim
        cout = z_dim / (2 ^ self.layer_size)
        for i in range(self.layer_size + 1):
            setattr(self, 'enc_' + str(i), nn.Sequential(
                nn.Conv2d(cin, cout, 4, 2, 1),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2)
            ))
            cin = cout
            cout = cout * 2

        cin = cin / 2
        cout = cout / 2
        setattr(self, 'enc_mu', nn.Sequential(
            nn.Conv2d(cin, cout, 4, 2, 1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2)))
        setattr(self, 'enc_logvar', nn.Sequential(
            nn.Conv2d(cin, cout, 4, 2, 1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2)))

        # Decoder
        cin = cout + c_dim
        setattr(self, 'dec_0', nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, 2, 1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2)))
        for i in range(self.layer_size):
            setattr(self, 'dec' + str(i + 1), nn.Sequential(
                nn.Conv2d(cin, cout, 4, 2, 1),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2)
            ))
            cin = cout
            cout = cout / 2
        cout = img_dim
        setattr(self, 'dec' + str(self.layer_size), nn.Sequential(
            nn.Conv2d(cin, cout, 4, 2, 1),
            nn.Tanh()))

        self.apply(weights_init)

    def encoder(self, x):
        for i in range(self.layer_size + 1):
            x = getattr(self, 'enc_' + str(i))(x)

        mu = getattr(self, 'enc_mu')(x)
        logvar = getattr(self, 'enc_logvar')(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, insert_attrs=None):
        for l in range(len(self.dec_layers)):
            if len(z.size()) != 4:
                z = z.unsqueeze(-1).unsqueeze(-1)
            if (insert_attrs is not None):
                if len(z.size()) == 2:
                    z = torch.cat([z, insert_attrs], dim=1)
                else:
                    H, W = z.size()[2], z.size()[3]
                    z = torch.cat([z, insert_attrs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)], dim=1)
            z = getattr(self, 'dec_' + str(l))(z)
        return z

    def forward(self, x, insert_attrs=None, return_enc=False):
        batch_size = x.size()[0]
        mu, logvar = self.encode(x)
        if len(mu.size()) > 2:
            mu = mu.view(batch_size, -1)
            logvar = logvar.view(batch_size, -1)
        z = self.reparameterize(mu, logvar)
        if return_enc:
            return z
        else:
            return self.decode(z, insert_attrs), mu, logvar


# Discriminator
class Discriminator_pixel(nn.Module):
    def __init__(self, img_dim, layer_size=5):
        super(Discriminator_pixel, self).__init__()
        cin = img_dim
        cout = 512 / (2 ^ layer_size)
        self.layer_size = layer_size
        for i in range(self.layer_size + 1):
            setattr(self, 'disp_' + str(i), nn.Sequential(
                nn.Conv2d(cin, cout, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ))
            cin = cout
            cout = cout * 2

        setattr(self, 'disp_' + str(self.layer_size + 1), nn.Sequential(
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0)))

        setattr(self, 'disp_out_1', nn.Sequential(
            nn.Linear(512, 1), nn.Dropout(0)
        ))
        setattr(self, 'disp_out_3', nn.Sequential(
            nn.Linear(512, 3), nn.Dropout(0)
        ))

        self.apply(weights_init)

    def forward(self, x):
        for i in range(self.layer_size + 1):
            x = getattr(self, 'disp_' + str(i))(x)

        # fc
        x = x.view(x.size()[0], -1)
        X = getattr(self, 'disp_' + str(self.layer_size + 1))(x)

        # out1+out3
        x1 = getattr(self, 'disp_out_1')(x)
        X3=getattr(self,'disp_out_3')(x)

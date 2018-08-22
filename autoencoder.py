import torch
import torch.nn as nn
from wct import wctransform
from encoder_decoder_vgg19 import Encoder, Decoder

class SingleLevelWCT(nn.Module):

    def __init__(self, args):
        super(SingleLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')  # on average svd takes 4604ms on cpu vs gpu 5312ms on a 512x512 content/591x800 style (comprehensive of data transferring)
        self.device = args.device
        self.alpha = args.alpha
        self.e5 = Encoder(5)
        self.d5 = Decoder(5)


    def forward(self, content_img, style_img):

        cf = self.e5(content_img).data.to(device=self.svd_device).squeeze(0)
        sf = self.e5(style_img).data.to(device=self.svd_device).squeeze(0)

        # csf = cf
        # csf = csf.unsqueeze(0)
        csf = wctransform(self.alpha, cf, sf)
        csf = csf.to(device=self.device)

        out = self.d5(csf)

        return out


class MultiLevelWCT(nn.Module):

    def __init__(self, args):
        super(MultiLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')
        self.device = args.device
        self.alpha = args.alpha

        self.e1 = Encoder(1)
        self.e2 = Encoder(2)
        self.e3 = Encoder(3)
        self.e4 = Encoder(4)
        self.e5 = Encoder(5)
        self.encoders = [self.e5, self.e4, self.e3, self.e2, self.e1]

        self.d1 = Decoder(1)
        self.d2 = Decoder(2)
        self.d3 = Decoder(3)
        self.d4 = Decoder(4)
        self.d5 = Decoder(5)
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]


    def forward(self, content_img, style_img):

        def stylize_wct(l, content_f, style_f):
            cf = self.encoders[l](content_f).data.to(device=self.svd_device).squeeze(0)
            sf = self.encoders[l](style_f).data.to(device=self.svd_device).squeeze(0)
            csf = wctransform(self.alpha, cf, sf).to(device=self.device)
            return self.decoders[l](csf)

        for i in range(len(self.encoders)):
            content_img = stylize_wct(i, content_img, style_img)

        return content_img
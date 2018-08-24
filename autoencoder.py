import torch
import torch.nn as nn
from feature_transform import wc_transform as wct
from encoder_decoder_vgg19 import Encoder, Decoder


def stylize_wct(level, content, style0, encoders, decoders, alpha, svd_device, cnn_device, style1=None, beta=None):
    with torch.no_grad():
        cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
        s0f = encoders[level](style0).data.to(device=svd_device).squeeze(0)
        if beta:
            s1f = encoders[level](style1).data.to(device=svd_device).squeeze(0)
            csf = wct(alpha, cf, s0f, s1f, beta).to(device=cnn_device)
        else:
            csf = wct(alpha, cf, s0f).to(device=cnn_device)
        return decoders[level](csf)

class SingleLevelWCT(nn.Module):

    def __init__(self, args):
        super(SingleLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')  # on average svd takes 4604ms on cpu vs gpu 5312ms on a 512x512 content/591x800 style (comprehensive of data transferring)
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta

        self.e5 = Encoder(5)
        self.encoders = [self.e5]
        self.d5 = Decoder(5)
        self.decoders = [self.d5]

    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):

        if additional_style_flag:
            out = stylize_wct(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                              self.cnn_device, style1=style_img1, beta=self.beta)
        else:
            out = stylize_wct(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                              self.cnn_device)

        return out


class MultiLevelWCT(nn.Module):

    def __init__(self, args):
        super(MultiLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta

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

    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):

        for i in range(len(self.encoders)):
            if additional_style_flag:
                content_img = stylize_wct(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                                  self.cnn_device, style1=style_img1, beta=self.beta)
            else:
                content_img = stylize_wct(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                                  self.cnn_device)

        return content_img
import torch.nn as nn
from wct import wctransform
from encoder_decoder_vgg19 import Encoder, Decoder

class SingleLevelWCT(nn.Module):

    def __init__(self, args):
        super(SingleLevelWCT, self).__init__()

        self.alpha = args.alpha
        self.e5 = Encoder(5)
        self.d5 = Decoder(5)


    def forward(self, content_img, style_img):

        cf = self.e5(content_img).data.cpu().squeeze(0)
        sf = self.e5(style_img).data.cpu().squeeze(0)

        csf = cf
        # csf = wctransform(self.alpha, cf, sf)

        csf = csf.unsqueeze(0)
        out = self.d5(csf)

        return out


# class MultiLevelWCT(nn.Module):
#
#     def __init__(self, args):
#         super(MultiLevelWCT, self).__init__()
#
#         self.alpha = args.alpha
#
#         e1_vgg1 = load_lua(args.encoder1_vgg19)
#         e2_vgg2 = load_lua(args.encoder2_vgg19)
#         e3_vgg3 = load_lua(args.encoder3_vgg19)
#         e4_vgg4 = load_lua(args.encoder4_vgg19)
#         e5_vgg5 = load_lua(args.encoder5_vgg19)
#
#         self.e1 = Encoder(e1_vgg1, 1)
#         self.e2 = Encoder(e2_vgg2, 2)
#         self.e3 = Encoder(e3_vgg3, 3)
#         self.e4 = Encoder(e4_vgg4, 4)
#         self.e5 = Encoder(e5_vgg5, 5)
#
#         d1_vgg1 = load_lua(args.decoder1_vgg19)
#         d2_vgg2 = load_lua(args.decoder2_vgg19)
#         d3_vgg3 = load_lua(args.decoder3_vgg19)
#         d4_vgg4 = load_lua(args.decoder4_vgg19)
#         d5_vgg5 = load_lua(args.decoder5_vgg19)
#
#         self.d1 = Decoder1(d1_vgg1)
#         self.d2 = Decoder2(d2_vgg2)
#         self.d3 = Decoder3(d3_vgg3)
#         self.d4 = Decoder4(d4_vgg4)
#         self.d5 = Decoder5(d5_vgg5)
#
#
#     def forward(self, content_img, style_img):
#
#         cf5 = self.e5(content_img).data.cpu().squeeze(0)
#         sf5 = self.e5(style_img).data.cpu().squeeze(0)
#         csf5 = wctransform(self.alpha, cf5, sf5)
#         im5 = self.d5(csf5)
#
#         cf4 = self.e4(im5).data.cpu().squeeze(0)
#         sf4 = self.e4(style_img).data.cpu().squeeze(0)
#         csf4 = wctransform(self.alpha, cf4, sf4)
#         im4 = self.d4(csf4)
#
#         cf3 = self.e3(im4).data.cpu().squeeze(0)
#         sf3 = self.e3(style_img).data.cpu().squeeze(0)
#         csf3 = wctransform(self.alpha, cf3, sf3)
#         im3 = self.d3(csf3)
#
#         cf2 = self.e2(im3).data.cpu().squeeze(0)
#         sf2 = self.e2(style_img).data.cpu().squeeze(0)
#         csf2 = wctransform(self.alpha, cf2, sf2)
#         im2 = self.d2(csf2)
#
#         cf1 = self.e1(im2).data.cpu().squeeze(0)
#         sf1 = self.e1(style_img).data.cpu().squeeze(0)
#         csf1 = wctransform(self.alpha, cf1, sf1)
#         im1 = self.d5(csf1)
#
#         out = im1
#         return out
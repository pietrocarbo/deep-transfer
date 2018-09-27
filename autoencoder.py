import PIL
import torch
from PIL import Image
import torch.nn as nn
from log_utils import get_logger
from feature_transforms import wct, wct_mask
from encoder_decoder_factory import Encoder, Decoder
import torchvision.transforms.functional as transforms


log = get_logger()


def stylize(level, content, style0, encoders, decoders, alpha, svd_device, cnn_device, interpolation_beta=None, style1=None, mask_mode=None, mask=None):
    log.debug('Stylization up to ReLu' + str(level) + ' of content sized: ' + str(content.size()) + ' and style sized: ' + str(style0.size()))

    with torch.no_grad():
        if mask_mode:
            cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
            s0f = encoders[level](style0).data.to(device=svd_device).squeeze(0)
            s1f = encoders[level](style1).data.to(device=svd_device).squeeze(0)
            log.debug('mask-mode: content features size: ' + str(cf.size()) + ', style 0 features size: ' + str(s0f.size()) + ', style 1 features size: ' + str(s1f.size()))

            cf_channels, cf_width, cf_height = cf.size(0), cf.size(1), cf.size(2)
            mask = transforms.to_tensor(transforms.resize(mask, (cf_height, cf_width), interpolation=PIL.Image.NEAREST))

            mask_view = mask.view(-1)
            mask_view = torch.gt(mask_view, 0.5)
            foreground_mask_ix = (mask_view == 1).nonzero().type(torch.LongTensor)
            background_mask_ix = (mask_view == 0).nonzero().type(torch.LongTensor)
            log.debug('mask-mode: ' + str((foreground_mask_ix.nelement() / mask_view.nelement()) * 100) + '% of the mask is foreground')

            cf_view = cf.view(cf_channels, -1)
            cf_fground_masked = torch.index_select(cf_view, 1, foreground_mask_ix.view(-1)).view(cf_channels, foreground_mask_ix.nelement())
            cf_bground_masked = torch.index_select(cf_view, 1, background_mask_ix.view(-1)).view(cf_channels, background_mask_ix.nelement())

            csf_fground = wct_mask(cf_fground_masked, s0f)
            csf_bground = wct_mask(cf_bground_masked, s1f)

            csf = torch.zeros_like(cf_view)
            csf.index_copy_(1, foreground_mask_ix.view(-1), csf_fground)
            csf.index_copy_(1, background_mask_ix.view(-1), csf_bground)
            csf = csf.view_as(cf)

            csf = alpha * csf + (1.0 - alpha) * cf
            csf = csf.unsqueeze(0).to(device=cnn_device)

        elif interpolation_beta:
            cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
            s0f = encoders[level](style0).data.to(device=svd_device).squeeze(0)
            s1f = encoders[level](style1).data.to(device=svd_device).squeeze(0)
            log.debug('interpolation-mode: content features size: ' + str(cf.size()) + ', style 0 features size: ' + str(s0f.size()) + ', style 1 features size: ' + str(s1f.size()))

            csf = wct(alpha, cf, s0f, s1f, interpolation_beta).to(device=cnn_device)

        else:
            cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
            s0f = encoders[level](style0).data.to(device=svd_device).squeeze(0)
            log.debug('transfer-mode: content features size: ' + str(cf.size()) + ', style features size: ' + str(s0f.size()))

            csf = wct(alpha, cf, s0f).to(device=cnn_device)

        return decoders[level](csf)

class SingleLevelWCT(nn.Module):

    def __init__(self, args):
        super(SingleLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')  # on average svd takes 4604ms on cpu vs gpu 5312ms on a 512x512 content/591x800 style (comprehensive of data transferring)
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta

        if args.mask:
            self.mask_mode = True
            self.mask = Image.open(args.mask).convert('1')
        else:
            self.mask_mode = False
            self.mask = None


        self.e5 = Encoder(5)
        self.encoders = [self.e5]
        self.d5 = Decoder(5)
        self.decoders = [self.d5]

    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):

        if additional_style_flag:
            out = stylize(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                          self.cnn_device, interpolation_beta=self.beta, style1=style_img1, mask_mode=self.mask_mode, mask=self.mask)
        else:
            out = stylize(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                          self.cnn_device)

        return out


class MultiLevelWCT(nn.Module):

    def __init__(self, args):
        super(MultiLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta

        if args.mask:
            self.mask_mode = True
            self.mask = Image.open(args.mask).convert('1')
        else:
            self.mask_mode = False
            self.mask = None

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
                content_img = stylize(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                                      self.cnn_device, interpolation_beta=self.beta, style1=style_img1, mask_mode=self.mask_mode, mask=self.mask)
            else:
                content_img = stylize(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                                      self.cnn_device)

        return content_img
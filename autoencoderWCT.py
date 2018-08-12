import torch.nn as nn
from torch.utils.serialization import load_lua
from encoder_decoder_VGG19 import Encoder, Decoder1, Decoder2, Decoder3, Decoder4, Decoder5

class SingleLevelWCT(nn.Module):

    def __init__(self, args):
        super(SingleLevelWCT, self).__init__()

        e5_vgg5 = load_lua(args.encoder5_vgg19)
        self.e5 = Encoder(e5_vgg5, 5)

        d5_vgg5 = load_lua(args.decoder5_vgg19)
        self.d5 = Decoder5(d5_vgg5)


    def forward(self, content_img, style_img):
        out = self.e5(content_img)

        # def whiten_and_color(self,cF,sF):
        #     cFSize = cF.size()
        #     c_mean = torch.mean(cF,1) # c x (h x w)
        #     c_mean = c_mean.unsqueeze(1).expand_as(cF)
        #     cF = cF - c_mean
        #
        #     contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
        #     c_u,c_e,c_v = torch.svd(contentConv,some=False)
        #
        #     k_c = cFSize[0]
        #     for i in range(cFSize[0]):
        #         if c_e[i] < 0.00001:
        #             k_c = i
        #             break
        #
        #     sFSize = sF.size()
        #     s_mean = torch.mean(sF,1)
        #     sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        #     styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        #     s_u,s_e,s_v = torch.svd(styleConv,some=False)
        #
        #     k_s = sFSize[0]
        #     for i in range(sFSize[0]):
        #         if s_e[i] < 0.00001:
        #             k_s = i
        #             break
        #
        #     c_d = (c_e[0:k_c]).pow(-0.5)
        #     step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        #     step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        #     whiten_cF = torch.mm(step2,cF)
        #
        #     s_d = (s_e[0:k_s]).pow(0.5)
        #     targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
        #     targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        #     return targetFeature
        #
        # def transform(self,cF,sF,csF,alpha):
        #     cF = cF.double()
        #     sF = sF.double()
        #     C,W,H = cF.size(0),cF.size(1),cF.size(2)
        #     _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        #     cFView = cF.view(C,-1)
        #     sFView = sF.view(C,-1)
        #
        #     targetFeature = self.whiten_and_color(cFView,sFView)
        #     targetFeature = targetFeature.view_as(cF)
        #     ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        #     ccsF = ccsF.float().unsqueeze(0)
        #     csF.data.resize_(ccsF.size()).copy_(ccsF)
        #     return csF

        out = self.d5(out)
        return out


class MultiLevelWCT(nn.Module):

    def __init__(self, args):
        super(MultiLevelWCT, self).__init__()

        e1_vgg1 = load_lua(args.encoder1_vgg19)
        e2_vgg2 = load_lua(args.encoder2_vgg19)
        e3_vgg3 = load_lua(args.encoder3_vgg19)
        e4_vgg4 = load_lua(args.encoder4_vgg19)
        e5_vgg5 = load_lua(args.encoder5_vgg19)

        self.e1 = Encoder(e1_vgg1, 1)
        self.e2 = Encoder(e2_vgg2, 2)
        self.e3 = Encoder(e3_vgg3, 3)
        self.e4 = Encoder(e4_vgg4, 4)
        self.e5 = Encoder(e5_vgg5, 5)

        d1_vgg1 = load_lua(args.decoder1_vgg19)
        d2_vgg2 = load_lua(args.decoder2_vgg19)
        d3_vgg3 = load_lua(args.decoder3_vgg19)
        d4_vgg4 = load_lua(args.decoder4_vgg19)
        d5_vgg5 = load_lua(args.decoder5_vgg19)

        self.d1 = Decoder1(d1_vgg1)
        self.d2 = Decoder2(d2_vgg2)
        self.d3 = Decoder3(d3_vgg3)
        self.d4 = Decoder4(d4_vgg4)
        self.d5 = Decoder5(d5_vgg5)


    def forward(self, content_img, style_img):
        out = None


        return out
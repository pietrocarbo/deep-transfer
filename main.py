import os
import argparse
import PairDataset
import torchvision
from imshow_utils import *
import autoencoderWCT
from log_utils import get_logger
from torch.utils.data import DataLoader

log = get_logger()
folder = False

# batch size, #workers, content/style resizes, autoencoders weights paths, beta to balance texture synthesis
def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform',
                                     epilog='The VGG19 encoder-decoder weights .t7 files MUST be in the directory ./models')

    parser.add_argument('--content', help='path of the content image(s) to be trasformed')
    parser.add_argument('--style', help='path of the style image(s) to use')

    parser.add_argument('--contentDir', help='path of the directory containing the content images to be trasformed')
    parser.add_argument('--styleDir', help='path of the directory containing the style images to be used')

    parser.add_argument('--contentSize', type=int, help='New (minimum) size for the content image. To keep the original size set to 0') # default=768 in the paper
    parser.add_argument('--styleSize', type=int, help='New (minimum) size for the style image. To keep the original size set to 0')

    parser.add_argument('--autoencoder1-vgg19', default='models/autoencoder_vgg19/vgg19_1', help='Path to the folder containing .py files (models definition) and .pth files (weights) of VGG19 encoder and decoder upto conv1_1')
    parser.add_argument('--autoencoder2-vgg19', default='models/autoencoder_vgg19/vgg19_2', help='Path to the folder containing .py files (models definition) and .pth files (weights) of VGG19 encoder and decoder upto conv2_1')
    parser.add_argument('--autoencoder3-vgg19', default='models/autoencoder_vgg19/vgg19_3', help='Path to the folder containing .py files (models definition) and .pth files (weights) of VGG19 encoder and decoder upto conv3_1')
    parser.add_argument('--autoencoder4-vgg19', default='models/autoencoder_vgg19/vgg19_4', help='Path to the folder containing .py files (models definition) and .pth files (weights) of VGG19 encoder and decoder upto conv4_1')
    parser.add_argument('--autoencoder5-vgg19', default='models/autoencoder_vgg19/vgg19_5', help='Path to the folder containing .py files (models definition) and .pth files (weights) of VGG19 encoder and decoder upto conv5_1')

    parser.add_argument('--outDir', default='./outputs', help='path of the directory where stylized results will be saved')

    parser.add_argument('--alpha', type=float, default=0.6, help='hyperparameter controlling the blending of WCT features and content features')

    parser.add_argument('--cuda', default='false', action='store_true', help='Flag to enables GPU (CUDA) accelerated computations')

    return parser.parse_args()


def validate_args(args):
    supported_img_formats = ('.png', '.jpg', '.jpeg')

    assert((args.content and args.style) or (args.contentDir and args.styleDir))
    if args.content and args.style:
        ok = os.path.isfile(args.content) and os.path.isfile(args.content)
        ok &= os.path.splitext(args.content)[1].lower().endswith(supported_img_formats)
        ok &= os.path.splitext(args.style)[1].lower().endswith(supported_img_formats)
        if not ok: raise ValueError('content and style must be existing image paths')
    else:
        global folder
        folder = True
        if not os.path.isdir(args.contentDir) or not os.path.isdir(args.styleDir):
            raise ValueError('contentDir and styleDir must be existing directory paths')
        ok = any([os.path.splitext(file)[1].lower().endswith(supported_img_formats) for file in os.listdir(args.contentDir)])
        ok &= any([os.path.splitext(file)[1].lower().endswith(supported_img_formats) for file in os.listdir(args.styleDir)])
        if not ok: raise ValueError('contentDir and styleDir must contain at least one image file')

    if not 0. < args.alpha < 1.:
        raise ValueError('alpha value MUST be between 0 and 1')

    return args


def main():
    args = validate_args(parse_args())

    try:
        os.makedirs(args.outDir, exist_ok=True)
    except Exception:
        log.exception('Error encoutered while creating output directory')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    cspd = PairDataset.ContentStylePairDataset(args.content, args.style, content_transforms=transforms, style_transforms=transforms) if not folder \
        else PairDataset.ContentStylePairDataset(args.contentDir, args.styleDir, content_transforms=transforms, style_transforms=transforms)
    cspd_loader = DataLoader(cspd, batch_size=1, shuffle=False, num_workers=0)

    for sample in cspd_loader:
        model = autoencoderWCT.SingleLevelWCT(args)
        model.eval()
        out = model(sample['content'], sample['style'])
        tensor_imshow(out.detach().squeeze(0))


if __name__ == "__main__":
    # execute only if run as a script
    main()

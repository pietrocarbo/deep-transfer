import os
import re
import torch
import argparse
import PairDataset
import TripletDataset
import torchvision
import autoencoder
from log_utils import get_logger
from torch.utils.data import DataLoader

log = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform',
                                     epilog='Supported image file formats are: jpg, jpeg, png')

    parser.add_argument('--content', help='Path of the content image (or a directory containing images) to be trasformed')
    parser.add_argument('--style', help='Path of the style image (or a directory containing images) to use')
    parser.add_argument('--synthesis', default=False, action='store_true', help='Flag to syntesize a new texture. Must provide a texture style image')
    parser.add_argument('--stylePair', help='Path of two style images (separated by ",") to use in combination')
    parser.add_argument('--mask', help='Path of the binary mask image (white on black) to trasfer the style pair in the corrisponding areas')

    parser.add_argument('--contentSize', type=int, help='Reshape content image to have the new specified maximum size (keeping aspect ratio)') # default=768 in the paper
    parser.add_argument('--styleSize', type=int, help='Reshape style image to have the new specified maximum size (keeping aspect ratio)')

    parser.add_argument('--outDir', default='outputs', help='Path of the directory where stylized results will be saved')
    parser.add_argument('--outPrefix', help='Name prefixed in the saved stylized images')

    parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the blending between original content features and WCT-transformed features')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyperparameter balancing the interpolation between the two images in the stylePair')
    parser.add_argument('--no-cuda', default=False, action='store_true', help='Flag to enables GPU (CUDA) accelerated computations')
    parser.add_argument('--single-level', default=False, action='store_true', help='Flag to switch to single level stylization')

    return parser.parse_args()


def validate_args(args):
    supported_img_formats = ('.png', '.jpg', '.jpeg')

    # assert that we have a combinations of cli args meaningful to perform some task
    assert((args.content and args.style)   or (args.content and args.stylePair) or (args.style and args.synthesis) or (args.stylePair and args.synthesis) or (args.mask and args.content and args.stylePair))

    if args.content:
        if os.path.isfile(args.content) and os.path.splitext(args.content)[-1].lower().endswith(supported_img_formats):
            pass
        elif os.path.isdir(args.content) and any([os.path.splitext(file)[-1].lower().endswith(supported_img_formats) for file in os.listdir(args.content)]):
            pass
        else:
            raise ValueError("--content '" + args.content + "' must be an existing image file or a directory containing at least one supported image")

    if args.style:
        if os.path.isfile(args.style) and os.path.splitext(args.style)[-1].lower().endswith(supported_img_formats):
            pass
        elif os.path.isdir(args.style) and any([os.path.splitext(file)[-1].lower().endswith(supported_img_formats) for file in os.listdir(args.style)]):
            pass
        else:
            raise ValueError("--style '" + args.style + "' must be an existing image file or a directory containing at least one supported image")

    if args.stylePair:
        if len(args.stylePair.split(',')) == 2:
            args.style0 = args.stylePair.split(',')[0]
            args.style1 = args.stylePair.split(',')[1]
            if os.path.isfile(args.style0) and os.path.splitext(args.style0)[-1].lower().endswith(supported_img_formats) and \
                    os.path.isfile(args.style1) and os.path.splitext(args.style1)[-1].lower().endswith(supported_img_formats):
                pass
            else:
                raise ValueError("--stylePair '" + args.stylePair + "' must be an existing and supported image file paths pair")
            pass
        else:
            raise ValueError('--stylePair must be a comma separeted pair of image file paths')

    if args.mask:
        if os.path.isfile(args.mask) and os.path.splitext(args.mask)[-1].lower().endswith(supported_img_formats):
            pass
        else:
            raise ValueError("--mask '" + args.mask + "' must be an existing and supported image file path")

    if args.outDir != './outputs':
        args.outDir = os.path.normpath(args.outDir)
        if re.search(r'[^A-Za-z0-9- :_\\\/]', args.outDir):
            raise ValueError("--outDir '" + args.outDir + "' contains illegal characters")

    if args.outPrefix:
        args.outPrefix = os.path.normpath(args.outPrefix)
        if re.search(r'[^A-Za-z0-9-_\\\/]', args.outPrefix):
            raise ValueError("--outPrefix '" + args.outPrefix + "' contains illegal characters")

    if args.contentSize and (args.contentSize <= 0 or args.contentSize > 3840):
        raise ValueError("--contentSize '" + args.contentSize + "' have an invalid value (must be between 0 and 3840)")

    if args.styleSize and (args.styleSize <= 0 or args.styleSize > 3840):
        raise ValueError("--styleSize '" + args.styleSize + "' have an invalid value (must be between 0 and 3840)")

    if not 0. <= args.alpha <= 1.:
        raise ValueError("--alpha '" + args.alpha + "' have an invalid value (must be between 0 and 1)")

    if not 0. <= args.beta <= 1.:
        raise ValueError("--beta '" + args.beta + "' have an invalid value (must be between 0 and 1)")

    return args


def save_image(img, content_name, style_name, out_ext, args):
    torchvision.utils.save_image(img.cpu().detach().squeeze(0),
     os.path.join(args.outDir,
      (args.outPrefix + '_' if args.outPrefix else '') + content_name + '_stylized_by_' + style_name + '_alpha_' + str(int(args.alpha*100)) + '.' + out_ext))


def main():
    args = validate_args(parse_args())

    try:
        os.makedirs(args.outDir, exist_ok=True)
    except OSError:
        log.exception('Error encoutered while creating output directory ' + args.outDir)

    if not args.no_cuda and torch.cuda.is_available():
        log.info('Utilizing the first CUDA gpu available')
        args.device = torch.device('cuda:0')
    else:
        log.info('Utilizing the cpu for computations')
        args.device = torch.device('cpu')

    if args.synthesis:  args.alpha = 1.0

    if args.stylePair:
        log.info('Creating content and (two) styles triplets dataset object')
        dataset = TripletDataset.ContentStyleTripletDataset(args)
    else:
        log.info('Creating content and style pairs dataset object')
        dataset = PairDataset.ContentStylePairDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.single_level:
        log.info('Using single-level stylization pipeline')
        model = autoencoder.SingleLevelWCT(args)
    else:
        log.info('Using multi-level stylization pipeline')
        model = autoencoder.MultiLevelWCT(args)
    model.to(device=args.device)
    model.eval()

    for i, sample in enumerate(dataloader):
        log.info('Starting ' + str(i) + '/'+ str(len(dataloader)) + ' stylization iteration')

        if args.stylePair:
            log.info('content: ' + str(sample['contentPath']) + '\tstyle 1: ' + str(sample['style0Path']) + '\tstyle 2: ' + str(sample['style1Path']))

            s0_basename = str(os.path.basename(sample['style0Path'][0]).split('.')[0])
            s0_ext = str(os.path.basename(sample['style0Path'][0]).split('.')[-1])
            s1_basename = str(os.path.basename(sample['style1Path'][0]).split('.')[0])

            content = sample['content'].to(device=args.device)
            style0 = sample['style0'].to(device=args.device)
            style1 = sample['style1'].to(device=args.device)

            if args.synthesis:
                for ii in range(1, 4):
                    content = model(content, style0, True, style1)
                    save_image(content, 'texture_iter' + str(ii), s0_basename + '_and_' + s1_basename, str(s0_ext), args)
            else:
                c_basename = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
                c_ext = str(os.path.basename(sample['contentPath'][0]).split('.')[-1])

                out = model(content, style0, True, style1)
                save_image(out, c_basename, s0_basename + '_and_' + s1_basename, c_ext, args)
        else:
            log.info('content: ' + str(sample['contentPath']) + '\tstyle: ' + str(sample['stylePath']))

            s_basename = str(os.path.basename(sample['stylePath'][0]).split('.')[0])
            s_ext = str(os.path.basename(sample['stylePath'][0]).split('.')[-1])

            content = sample['content'].to(device=args.device)
            style = sample['style'].to(device=args.device)

            if args.synthesis:
                for ii in range(1, 4):
                    content = model(content, style)
                    save_image(content, 'texture_iter' + str(ii), s_basename, s_ext, args)
            else:
                c_basename = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
                c_ext = str(os.path.basename(sample['contentPath'][0]).split('.')[-1])

                from timeit import default_timer as timer
                start = timer()
                out = model(content, style)
                end = timer()
                log.info('Wall-clock time took for stylization: ' + str(end - start) + 's')
                # save_image(out, c_basename, s_basename, c_ext, args)

    log.info('Stylization completed, exiting.')

if __name__ == "__main__":
    main()

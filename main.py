import os
import argparse
import logging


def log_config():
    console_logs_lvl = file_logs_lvl = logging.INFO  # CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    console_logs_format = file_logs_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(__name__)
    logger.setLevel(console_logs_lvl)
    logging.basicConfig(format=console_logs_format)

    handler = logging.FileHandler('logs.txt', mode='w')
    handler.setLevel(file_logs_lvl)
    handler.setFormatter(logging.Formatter(file_logs_format))
    logger.addHandler(handler)

    return logger


log = log_config()


# batch size, #workers, content/style resizes, autoencoders weights paths, beta to balance texture synthesis
def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform',
                                     epilog='The VGG19 encoder-decoder weights .t7 files MUST be in the directory ./models')

    parser.add_argument('--content', help='path of the content image(s) to be trasformed')
    parser.add_argument('--style', help='path of the style image(s) to use')

    parser.add_argument('--contentDir', help='path of the directory containing the content images to be trasformed')
    parser.add_argument('--styleDir', help='path of the directory containing the style images to be used')

    parser.add_argument('--outDir', default='./', help='path of the directory where stylized results will be saved')

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
        if not os.path.isdir(args.contentDir) or not os.path.isdir(args.styleDir):
            raise ValueError('contentDir and styleDir must be existing directory paths')
        ok = any([os.path.splitext(file)[1].lower().endswith(supported_img_formats) for file in os.listdir(args.contentDir)])
        ok &= any([os.path.splitext(file)[1].lower().endswith(supported_img_formats) for file in os.listdir(args.styleDir)])
        if not ok: raise ValueError('contentDir and styleDir must contain at least one image file')

    if 0. < args.alpha < 1.:
        raise ValueError('alpha value MUST be between 0 and 1')

    return args


def main():
    args = validate_args(parse_args())

    try:
        os.makedirs(args.outDir)
    except Exception:
        log.exception('Error encoutered while creating output directory')


if __name__ == "__main__":
    # execute only if run as a script
    exit(main())

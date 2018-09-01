import os
import torch
from log_utils import get_logger
from im_utils import load_img
from torch.utils.data import Dataset

log = get_logger()
supported_img_formats = ('.png', '.jpg', '.jpeg')

class ContentStylePairDataset(Dataset):

    def __init__(self, args):
        super(Dataset, self).__init__()

        self.synthesis = args.synthesis
        self.contentSize = args.contentSize
        self.styleSize = args.styleSize

        if args.style.endswith(supported_img_formats):
            self.pairs_fn = [('texture', args.style) if args.synthesis else (args.content, args.style)]
        else:
            self.pairs_fn = []
            for c in os.listdir(args.content):
                for s in os.listdir(args.style):
                    path_pair = (os.path.join(args.content, ('texture' if args.synthesis else c)), os.path.join(args.style, s))
                    log.info('Adding: ' + str(path_pair) + ' to the dataset')
                    self.pairs_fn.append(path_pair)

    def __len__(self):
        return len(self.pairs_fn)

    def __getitem__(self, idx):
        pair = self.pairs_fn[idx]

        style = load_img(pair[1], self.styleSize)

        if self.synthesis:
            c_c, h_c, w_c = style.size()
            content = torch.zeros((c_c, h_c, w_c)).uniform_()
        else:
            content = load_img(pair[0], self.contentSize)

        return {'content': content, 'contentPath': pair[0], 'style': style, 'stylePath': pair[1]}
import os
import torch
from im_utils import load_img
from log_utils import get_logger
from torch.utils.data import Dataset

log = get_logger()
supported_img_formats = ('.png', '.jpg', '.jpeg')

class ContentStyleTripletDataset(Dataset):

    def __init__(self, args):
        super(Dataset, self).__init__()

        self.synthesis = args.synthesis
        self.contentSize = args.contentSize
        self.styleSize = args.styleSize

        if args.synthesis:
            self.triplets_fn = [('texture', args.style0, args.style1)]
        elif args.content and args.content.endswith(supported_img_formats):
            self.triplets_fn = [(args.content, args.style0, args.style1)]
        else:
            self.triplets_fn = []
            for c in os.listdir(args.content):
                path_triplet = (os.path.join(args.content, c), args.style0, args.style1)
                log.info('Adding: ' + str(path_triplet) + ' to the dataset')
                self.triplets_fn.append(path_triplet)

    def __len__(self):
        return len(self.triplets_fn)

    def __getitem__(self, idx):
        triplet = self.triplets_fn[idx]

        style0 = load_img(triplet[1], self.styleSize)
        style1 = load_img(triplet[2], self.styleSize)

        if self.synthesis:
            c_c, h_c, w_c = style0.size()
            content = torch.zeros((c_c, h_c, w_c)).uniform_()
        else:
            content = load_img(triplet[0], self.contentSize)

        return {'content': content, 'contentPath': triplet[0], 'style0': style0, 'style0Path': triplet[1], 'style1': style1, 'style1Path': triplet[2]}
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class MetricsDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot)  # create a path '/path/to/data/trainA'
        paths = sorted(make_dataset(self.dir, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.paths=[]
        for path in paths:
            if path.__contains__('fake_A'):
                self.paths.append(path)
        random.shuffle(self.paths)
        self.size=len(self.paths)
        params = {'crop_pos': (144,144)}
        self.transform = get_transform(self.opt, params,grayscale=(self.opt.input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        path=self.paths[index % self.size]
        img=Image.open(path)
        img = self.transform(img)
        label=0
        return {'img': img, 'label': label, 'path': path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size

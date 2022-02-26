import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class MaskedDataset(BaseDataset):
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
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_A_mask = self.dir_A+'_mask'
        self.dir_A_point = self.dir_A+'_point'
        self.dir_B_mask = self.dir_B+'_mask'
        self.dir_B_point = self.dir_B+'_point'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.A_mask_paths=sorted(make_dataset(self.dir_A_mask, opt.max_dataset_size))
        self.A_point_paths=sorted(make_dataset(self.dir_A_point, opt.max_dataset_size))
        self.B_mask_paths=sorted(make_dataset(self.dir_B_mask, opt.max_dataset_size))
        self.B_point_paths=sorted(make_dataset(self.dir_B_point, opt.max_dataset_size))

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_mask_path=self.A_mask_paths[index % self.A_size]
        A_point_path=self.A_point_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_mask_path = self.B_mask_paths[index_B]
        B_point_path = self.B_point_paths[index_B]
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # open masks
        A_mask= self.transform_A(Image.open(A_mask_path))
        B_mask = self.transform_A(Image.open(B_mask_path))
        # read points
        A_point=[]
        with open(A_point_path) as f:
            for line in f.readlines():
                x,y,_,_=list(map(int,line.split(',')))
                A_point.append((x,y))
        B_point=[]
        with open(B_point_path) as f:
            for line in f.readlines():
                x,y,_,_=list(map(int,line.split(',')))
                B_point.append((x,y))

        return {'A': A, 'B': B, 'A_mask': A_mask, 'A_point': A_point, 'A_paths': A_path, 'B_mask': B_mask, 'B_point': B_point, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

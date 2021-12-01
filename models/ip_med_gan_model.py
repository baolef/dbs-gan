import torch

from . import networks
from .base_model import BaseModel
from ignite.metrics import *
from util.loss import StyleLoss, PercepLoss

class IpMedGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_Dg', type=float, default=0.8, help='weight for global discriminator loss')
        parser.add_argument('--lambda_Dl', type=float, default=0.2, help='weight for local discriminator loss')
        parser.add_argument('--lambda_style', type=float, default=0.0001, help='weight for style loss')
        parser.add_argument('--lambda_percep', type=float, default=0.0001, help='weight for perception loss')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--n_layers_Dg', type=int, default=3, help='global discriminator layers')
            parser.add_argument('--n_layers_Dl', type=int, default=2, help='local discriminator layers')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN_global', 'G_GAN_local', 'G_L1', 'Dl', 'Dl_real', 'Dl_fake', 'Dg',
                           'Dg_real', 'Dg_fake','style','percep']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.metrics={'ssim': SSIM(2), 'psnr': PSNR(2), 'mse': MeanSquaredError(), 'mae': MeanAbsoluteError()}
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'Dg', 'Dl']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDg = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_Dg, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDl = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_Dl, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.vgg19 = networks.define_vgg19(opt.input_nc, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionStyle=StyleLoss(self.gpu_ids[0])
            self.criterionPercep =PercepLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dg = torch.optim.Adam(self.netDg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dl = torch.optim.Adam(self.netDl.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_Dg)
            self.optimizers.append(self.optimizer_Dl)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        mask = input['mask']
        self.left = mask['left'].int().numpy()
        self.right = mask['right'].int().numpy()
        self.size = mask['size'][0].int().numpy()

    def crop(self):
        real_A_crop = []
        real_B_crop = []
        fake_B_crop = []
        for i in range(len(self.real_A)):
            real_A_crop.append(self.real_A[i, :, self.left[i, 0]:self.left[i, 0] + self.size[0],
                               self.left[i, 1]:self.left[i, 1] + self.size[1]])
            real_A_crop.append(self.real_A[i, :, self.right[i, 0]:self.right[i, 0] + self.size[0],
                               self.right[i, 1]:self.right[i, 1] + self.size[1]])
            real_B_crop.append(self.real_B[i, :, self.left[i, 0]:self.left[i, 0] + self.size[0],
                               self.left[i, 1]:self.left[i, 1] + self.size[1]])
            real_B_crop.append(self.real_B[i, :, self.right[i, 0]:self.right[i, 0] + self.size[0],
                               self.right[i][1]:self.right[i, 1] + self.size[1]])
            fake_B_crop.append(self.fake_B[i, :, self.left[i, 0]:self.left[i, 0] + self.size[0],
                               self.left[i, 1]:self.left[i, 1] + self.size[1]])
            fake_B_crop.append(self.fake_B[i, :, self.right[i, 0]:self.right[i, 0] + self.size[0],
                               self.right[i, 1]:self.right[i, 1] + self.size[1]])

        self.real_A_crop = torch.stack(real_A_crop)
        self.real_B_crop = torch.stack(real_B_crop)
        self.fake_B_crop = torch.stack(fake_B_crop)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_Dg(self):
        """Calculate GAN loss for the global discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB_global = torch.cat((self.real_A, self.fake_B),
                                   1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake_global = self.netDg(fake_AB_global.detach())
        self.loss_Dg_fake = self.criterionGAN(pred_fake_global, False)
        # Real
        real_AB_global = torch.cat((self.real_A, self.real_B), 1)
        pred_real_global = self.netDg(real_AB_global)
        self.loss_Dg_real = self.criterionGAN(pred_real_global, True)
        # combine loss and calculate gradients
        self.loss_Dg = (self.loss_Dg_fake + self.loss_Dg_real) * 0.5
        self.loss_Dg.backward()

    def backward_Dl(self):
        """Calculate GAN loss for the local discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB_local = torch.cat((self.real_A_crop, self.fake_B_crop),
                                  1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake_local = self.netDl(fake_AB_local.detach())

        self.loss_Dl_fake = self.criterionGAN(pred_fake_local, False)
        # Real
        real_AB_local = torch.cat((self.real_A_crop, self.real_B_crop), 1)
        pred_real_local = self.netDl(real_AB_local)

        self.loss_Dl_real = self.criterionGAN(pred_real_local, True)
        # combine loss and calculate gradients
        self.loss_Dl = (self.loss_Dl_fake + self.loss_Dl_real) * 0.5
        self.loss_Dl.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB_global = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_global = self.netDg(fake_AB_global)
        self.loss_G_GAN_global = self.criterionGAN(pred_fake_global, True)

        fake_AB_local = torch.cat((self.real_A_crop, self.fake_B_crop), 1)
        pred_fake_local = self.netDl(fake_AB_local)
        self.loss_G_GAN_local = self.criterionGAN(pred_fake_local, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # Third, style loss
        self.loss_style=self.criterionStyle(self.fake_B,self.real_B)
        # Fourth, perception loss
        self.loss_percep=self.criterionPercep(self.real_A,self.real_B,self.fake_B,self.netDg)
        # combine loss and calculate gradients
        self.loss_G = self.opt.lambda_Dg*self.loss_G_GAN_global + self.opt.lambda_Dl * self.loss_G_GAN_local +self.opt.lambda_L1 * self.loss_G_L1 +self.opt.lambda_style* self.loss_style + self.opt.lambda_percep*self.loss_percep
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        self.crop()
        # update Dg
        self.set_requires_grad(self.netDg, True)  # enable backprop for D
        self.optimizer_Dg.zero_grad()  # set D's gradients to zero
        self.backward_Dg()  # calculate gradients for D
        self.optimizer_Dg.step()  # update D's weights
        # update Dl
        self.set_requires_grad(self.netDl, True)  # enable backprop for D
        self.optimizer_Dl.zero_grad()  # set D's gradients to zero
        self.backward_Dl()  # calculate gradients for D
        self.optimizer_Dl.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netDg, False)  # Dg requires no gradients when optimizing G
        self.set_requires_grad(self.netDl, False)  # Dl requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def compute_metrics(self):
        for name in self.metrics.keys():
            self.metrics[name].update((self.fake_B,self.real_B))

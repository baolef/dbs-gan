import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


def get_image_gradients(image):
    """Returns image gradients (dy, dx) for each color channel.
    Both output tensors have the same shape as the input: [b, c, h, w].
    Places the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    That means that dy will always have zeros in the last row,
    and dx will always have zeros in the last column.

    This can be used to implement the anisotropic 2-D version of the
    Total Variation formula:
        https://en.wikipedia.org/wiki/Total_variation_denoising
    (anisotropic is using l1, isotropic is using l2 norm)

    Arguments:
        image: Tensor with shape [b, c, h, w].
    Returns:
        Pair of tensors (dy, dx) holding the vertical and horizontal image
        gradients (1-step finite difference).
    Raises:
      ValueError: If `image` is not a 3D image or 4D tensor.
    """

    image_shape = image.shape

    if len(image_shape) == 3:
        # The input is a single image with shape [height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        dx = image[:, 1:, :] - image[:, :-1, :]  # pixel_dif2, f_v_1-f_v_2
        dy = image[1:, :, :] - image[:-1, :, :]  # pixel_dif1, f_h_1-f_h_2

    elif len(image_shape) == 4:
        # Return tensors with same size as original image
        # adds one pixel pad to the right and removes one pixel from the left
        right = F.pad(image, [0, 1, 0, 0])[..., :, 1:]
        # adds one pixel pad to the bottom and removes one pixel from the top
        bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :]

        # right and bottom have the same dimensions as image
        dx, dy = right - image, bottom - image

        # this is required because otherwise results in the last column and row having
        # the original pixels from the image
        dx[:, :, :, -1] = 0  # dx will always have zeros in the last column, right-left
        dy[:, :, -1, :] = 0  # dy will always have zeros in the last row,    bottom-top
    else:
        raise ValueError(
            'image_gradients expects a 3D [h, w, c] or 4D tensor '
            '[batch_size, c, h, w], not %s.', image_shape)

    return dy, dx


class AttenLoss(nn.Module):
    def __init__(self):
        super(AttenLoss, self).__init__()

    def forward(self, ground_truth: Tensor, predict: Tensor, mask: Tensor):
        mse_loss = F.mse_loss(ground_truth, predict)
        gtdy, gtdx = get_image_gradients(ground_truth * mask)
        preddy, preddx = get_image_gradients(predict * mask)
        gradient_loss = (F.l1_loss(gtdx, preddx) + F.l1_loss(gtdy, preddy)) / 2

        return mse_loss + gradient_loss


def gram_matrix(x):
    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrixFrobenius norm
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, device):
        super(StyleLoss, self).__init__()
        vgg19_model = models.vgg19(pretrained=True)
        for param in vgg19_model.parameters():
            param.requires_grad = False
        vgg19_model_new = list(vgg19_model.features.children())[:18]
        vgg19_model_new[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg19 = nn.Sequential(*vgg19_model_new).to(device)

    def forward(self, xhat, x):
        loss = 0
        i = 0
        for layer in self.vgg19.children():
            xhat = layer(xhat)
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                i += 1
                loss += (torch.norm(gram_matrix(xhat) - gram_matrix(x)) / (2 * (i + 1))) ** 2

        return loss / i


class PercepLoss(nn.Module):
    def __init__(self):
        super(PercepLoss, self).__init__()

    def forward(self, real_A, real_B, fake_B, discriminator):
        if isinstance(discriminator, nn.DataParallel):
            discriminator = discriminator.module

        loss = 0
        # print(discriminator)
        # discriminator=list(discriminator.children())
        discriminator=list(discriminator.children())[0]
        # print(discriminator)
        a=torch.cat((real_A,fake_B),dim=1)
        b=torch.cat((real_A,real_B),dim=1)
        # a = discriminator[0](torch.cat((xhat, y),dim=1))
        # b = discriminator[0](torch.cat((x, y))
        cnt = 0
        for layer in discriminator.children():
            a = layer(a)
            b = layer(b)
            if isinstance(layer, nn.LeakyReLU):
                cnt += 1
                loss += F.l1_loss(a, b)
        return loss / cnt

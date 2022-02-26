from models import networks
import hiddenlayer as hl
import torch
from torchinfo import summary


def visualize(model, name, size, additional_transforms=None):
    if additional_transforms is None:
        additional_transforms = []
    x = torch.zeros(size=size)
    transforms = [hl.transforms.Prune('Constant'), hl.transforms.Rename('InstanceNormalization', to='IN')] + additional_transforms + [
        hl.transforms.Fold("Pad > Conv > IN > Relu", "InBlock"),
        hl.transforms.Fold("Pad > Conv > IN", "InBlock"),
        hl.transforms.Fold("Pad > Conv > Tanh", "OutBlock"),
        hl.transforms.Fold("Conv > IN > Relu", "DownBlock"),
        hl.transforms.Fold("ConvTranspose > IN > Relu", "UpBlock"),
        hl.transforms.Fold("Conv > IN > LeakyRelu", "ConvBlock"),
        hl.transforms.Fold("Conv > LeakyRelu", "InBlock"),
        # hl.transforms.FoldDuplicates()
        ]
    # transforms = [
    #     hl.transforms.Fold("Pad > Conv > InstanceNormalization > Relu", "Input"),
    #     hl.transforms.Fold("Pad > Conv > Tanh", "Output"),
    #     hl.transforms.Fold("Conv > InstanceNormalization > Relu", "DownBlock"),
    #     hl.transforms.Fold("ConvTranspose > InstanceNormalization > Relu", "UpBlock"),
    #     hl.transforms.Fold("Conv > InstanceNormalization > LeakyRelu", "ConvBlock"),
    #     hl.transforms.Fold("Conv > LeakyRelu", "Input"),
    # ]
    # transforms += additional_transforms
    graph = hl.build_graph(model, x, transforms=transforms)
    graph.theme = hl.graph.THEMES['blue'].copy()
    graph.save(name, format='png')

    # summary(model, x.shape)
    # print(model)


if __name__ == '__main__':
    # generator = networks.define_G(1, 1, 64, "resnet_9blocks", "instance")
    # add_trans = [
    #     hl.transforms.Fold("Pad > Conv > IN > Relu > Pad > Conv > IN > Add",
    #                        "ResBlock", "ResBlock"),
    #     # hl.transforms.FoldDuplicates()
    # ]
    # visualize(generator, "visualization/generator", [1, 1, 512, 512], add_trans)
    #
    # discriminator_global = networks.define_D(1, 64, "n_layers", 3, "instance")
    # visualize(discriminator_global, "visualization/discriminator_global", [1, 1, 512, 512])
    #
    # discriminator_local = networks.define_D(1, 64, "n_layers", 2, "instance")
    # visualize(discriminator_local, "visualization/discriminator_local", [1, 1, 40, 40])

    res_block = networks.ResnetBlock(256, "reflect", torch.nn.InstanceNorm2d, False, True)
    visualize(res_block, "visualization/res_block", [1, 256, 128, 128])

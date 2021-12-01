import torch
from torch.utils.tensorboard import SummaryWriter

class Writer():
    def __init__(self, opt):
        self.writer=SummaryWriter(comment='_'+opt.name)
        self.size=opt.crop_size
        self.batch_size=opt.batch_size

    def add_models(self, nets):
        for name, net in nets.items():
            channels=2 if name.startswith("D") else 1
            self.writer.add_graph(net,torch.zeros((1,channels,self.size, self.size)))

    def add_results(self, images, global_step):
        imgs_list=list(images.values())
        imgs=torch.cat(imgs_list,-1)
        self.writer.add_images("results",imgs,global_step)

    def add_losses(self, losses, global_step):
        for name, loss in losses.items():
            self.writer.add_scalar(name,loss,global_step)

    def add_lr(self,lr,global_step):
        self.writer.add_scalar("learning_rate",lr,global_step)

    def close(self):
        self.writer.close()

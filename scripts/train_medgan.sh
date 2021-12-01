nohup python train.py \
--gpu_ids 2 \
--name dbs \
--model med_gan \
--dataroot /data/baole/dbs/patch_pix2pix/before/2d/AB \
--direction AtoB --load_size 512 \
--crop_size 512 \
--display_port 1236 \
--batch_size 16 \
--suffix {model}_{netG}_size_{load_size} \
--display_freq 6400 \
--print_freq 1600 \
--netG cas_unet_256 \
--netD n_layers \
--n_epochs 100 \
--n_epochs_decay 100 \
--no_flip \
> train_med_gan.log 2>&1 &
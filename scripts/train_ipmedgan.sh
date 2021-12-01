nohup python train.py \
--gpu_ids 2 \
--name dbs \
--model ip_med_gan \
--dataroot /data/baole/dbs/patch_pix2pix/before/2d/AB \
--direction AtoB --load_size 512 \
--crop_size 512 \
--display_port 1238 \
--batch_size 8 \
--suffix {model}_{netG}_size_{load_size} \
--display_freq 6400 \
--print_freq 1600 \
--netG cas_unet_256 \
--netD n_layers \
--n_epochs 100 \
--n_epochs_decay 100 \
--patience 20 \
--no_flip \
--lambda_style 100000 \
--lambda_percep 100 \
> train_ip_med_gan_2.log 2>&1 &
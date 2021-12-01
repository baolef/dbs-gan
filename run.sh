nohup python train.py \
--gpu_ids 0 \
--name dbs \
--model med_gan \
--dataroot /data/baole/dbs/patch_pix2pix/before/2d/AB \
--direction AtoB --load_size 512 \
--crop_size 512 \
--display_port 1235 \
--batch_size 64 \
--suffix {model}_{netG}_size_{load_size} \
--display_freq 6400 \
--print_freq 100 \
--netD n_layers \
--n_epochs 100 \
--n_epochs_decay 100 \
--no_flip \
> train.log 2>&1 &
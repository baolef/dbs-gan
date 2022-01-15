python test.py \
--gpu_ids 3 \
--dataroot /data/baole/dbs/patch_pix2pix/after/2d/AB \
--results_dir ./results/after \
--name dbs \
--model med_gan \
--direction AtoB \
--load_size 512 \
--crop_size 512 \
--batch_size 64 \
--suffix {model}_{netG}_size_{load_size} \
--netG unet_256 \
--netD n_layers \
--epoch best \
--no_flip
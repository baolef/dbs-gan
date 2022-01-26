nohup python train.py \
--gpu_ids 2 \
--name dbs \
--model ant_gan_patch \
--dataset masked \
--dataroot /data/baole/dbs_zhongnan/unaligned_electrode \
--direction AtoB \
--load_size 512 \
--crop_size 512 \
--display_port 1235 \
--batch_size 2 \
--suffix {model}_{netG}_size_{load_size} \
--display_freq 1280 \
--print_freq 640 \
--netG resnet_9blocks \
--netD n_layers \
--n_epochs 100 \
--n_epochs_decay 100 \
--patience -1 \
--lambda_identity 1 \
--lambda_mask 1500 \
--no_flip \
> logs/train_ant_gan_patch_resnet_9blocks_18.log 2>&1 &
python test.py \
--gpu_ids 0 \
--dataroot /data/baole/dbs_oasis_compare/unaligned \
--results_dir ./results/after_oasis_compare \
--name dbs \
--model ant_gan \
--dataset unaligned \
--checkpoints_dir checkpoints_oasis_compare \
--direction AtoB \
--load_size 512 \
--crop_size 512 \
--batch_size 16 \
--suffix {model}_{netG}_{n_downsampling}_{netD}_{n_layers_D}_size_{load_size} \
--netG resnet_9blocks \
--n_downsampling 2 \
--netD n_layers \
--n_layers_D 3 \
--epoch latest \
--lambda_identity 0 \
--lambda_mask 1000 \
--diff_A fake_B \
--diff_B real_A \
--diff \
--serial_batches \
--no_flip
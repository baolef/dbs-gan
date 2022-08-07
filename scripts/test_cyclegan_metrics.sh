python test.py \
--gpu_ids 0 \
--dataroot /data/baole/dbs_zhongnan/aligned_ring \
--results_dir ./results/experiments_crop \
--name dbs \
--model cycle_gan \
--dataset masked \
--direction AtoB \
--load_size 512 \
--crop_size 512 \
--batch_size 16 \
--suffix {model}_{netG}_size_{load_size} \
--netG resnet_9blocks \
--netD n_layers \
--epoch best \
--lambda_identity 1 \
--checkpoints_dir experiments \
--metrics \
--serial_batches \
--diff \
--no_flip
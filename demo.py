import os
import numpy as np
import cv2

root="/home/baole/dbs/pytorch-CycleGAN-and-pix2pix/results/after/dbs_cycle_gan_resnet_9blocks_size_512_lambda_A_10.0_lambda_B_10.0_lambda_identity_10.0/test_best/images"
out = root.replace("images","demo")
if not os.path.exists(out):
    os.mkdir(out)
for file in os.listdir(root):
    if file.endswith("real_A.png"):
        path_A = os.path.join(root,file)
        path_B=os.path.join(root,file.replace("real_A","fake_B"))
        path_AB=os.path.join(out, file.replace("_real_A",""))
        im_A = cv2.imread(path_A, 0)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_B = cv2.imread(path_B, 0)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)

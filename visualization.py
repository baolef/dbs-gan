import ants
import os

import numpy as np
from PIL import Image

result_path = "/home/baole/dbs/pytorch-CycleGAN-and-pix2pix/results/after/dbs_ant_gan_patch_resnet_9blocks_size_512_lambda_A_10.0_lambda_B_10.0_lambda_identity_0.0_lambda_mask_1000.0/test_best/images"
nii_path = "/data/baole/dbs_zhongnan/raw/after"
output_path = os.path.join(result_path.rsplit("/",1)[0],"nii")

if not os.path.exists(output_path):
    os.makedirs(output_path)

path_dict = {}
for file in os.listdir(result_path):
    if file.endswith("fake_B.png"):
        name, id, _, _ = file.rsplit("_", 3)
        if name in path_dict.keys():
            path_dict[name].append((int(id), os.path.join(result_path, file)))
        else:
            path_dict[name] = [(int(id), os.path.join(result_path, file))]

for name, paths in path_dict.items():
    img = ants.image_read(os.path.join(nii_path, name + ".nii"))
    img_array = img.numpy()
    for id, path in paths:
        slice = Image.open(path).__array__()[:,:,0]
        slice = np.transpose(slice,(1,0))/255.0

        img_array[:, :, id] = slice.__array__()
    new_img = ants.from_numpy(img_array)
    new_img.set_spacing(img.spacing)
    new_img.set_direction(img.direction)
    new_img.set_origin(img.origin)
    ants.image_write(new_img, os.path.join(output_path, name + ".nii"))
    print(name)

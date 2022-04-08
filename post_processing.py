import ants
import os

import numpy as np
from PIL import Image

result_path = "/home/baole/dbs/pytorch-CycleGAN-and-pix2pix/results/after_all/dbs_ant_gan_patch_resnet_9blocks_n_layers_3_size_512_lambda_A_10.0_lambda_B_10.0_lambda_identity_0.0_lambda_mask_5000.0/test_best/images"
nii_path = "/data/baole/dbs_zhongnan/raw/after"
pts_path = "/data/baole/dbs_zhongnan/unaligned/testA_point"
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
        if 55<=id<=70:
            original_slice = Image.open(path.replace("fake_B.png","real_A.png")).__array__()[:,:,0]
            original_slice = np.transpose(original_slice, (1,0))/255.0
            pts=[]
            with open(os.path.join(pts_path, name+"_"+str(id)+".txt")) as pts_file:
                for line in pts_file.readlines():
                    pts.append(list(map(int,line.split(","))))
            for pt in pts:
                x,y,_,r=pt
                for i in range(len(slice)):
                    for j in range(len(slice[0])):
                        if (i-x)**2+(j-y)**2<=r**2:
                            slice[i][j]=original_slice[i][j]
        img_array[:, :, id] = slice.__array__()
    new_img = ants.from_numpy(img_array)
    new_img.set_spacing(img.spacing)
    new_img.set_direction(img.direction)
    new_img.set_origin(img.origin)
    ants.image_write(new_img, os.path.join(output_path, name + ".nii"))
    print(name)

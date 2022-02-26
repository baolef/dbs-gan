import os

root_path="experiments"
for root,dirs,files in os.walk(root_path):
    for file in files:
        path=os.path.join(root, file)
        if file.endswith(".pth") and file.split("_")[0].isnumeric():
            os.remove(path)


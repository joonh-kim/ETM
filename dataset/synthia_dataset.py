import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import imageio
import cv2

class SYNTHIADataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 256), ignore_label=255, set='train', num_classes=13):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        if self.num_classes == 13:
            self.id_to_trainid = {3: 0, 4: 1, 2: 2, 15: 3, 9: 4, 6: 5, 1: 6,
                                  10: 7, 17: 8, 8: 9, 19: 10, 12: 11, 11: 12}
        else:
            raise NotImplementedError("Unavailable number of classes")

        for name in self.img_ids:
            img_file = osp.join(self.root, self.set, name)
            label_file = osp.join(self.root, "GT/LABELS/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.asarray(imageio.imread(datafiles["label"], format='PNG-FI'))[:, :, 0]
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = cv2.resize(label, self.crop_size, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        return image, label_copy, name


if __name__ == '__main__':
    dst = SYNTHIADataSet("working_directory/data/SYNTHIA", "./synthia_list/val.txt",
                         crop_size=(512, 256), ignore_label=255, set='val', num_classes=13)
    trainloader = data.DataLoader(dst, batch_size=1)
    all_imgs = 0.0
    for i, data in enumerate(trainloader):
        imgs, labels, name = data
        all_imgs += imgs
        print(i, len(trainloader))
    img_mean = all_imgs.squeeze().view(3, -1).mean(dim=1) / len(trainloader)
    print(img_mean)

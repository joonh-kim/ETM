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

class IDDDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 256), ignore_label=255, set='train', num_classes=13):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        if self.num_classes == 13:
            self.id_to_trainid = {0: 0, 3: 1, 29: 2, 25: 3, 24: 4, 32: 5, 33: 6,
                                  6: 7, 8: 8, 12: 9, 14: 10, 9: 11, 10: 12}
        elif self.num_classes == 18:
            self.id_to_trainid = {0: 0, 3: 1, 29: 2, 20: 3, 21: 4, 26: 5, 25: 6,
                                  24: 7, 32: 8, 33: 9, 6: 10, 8: 11, 12: 12,
                                  13: 13, 14: 14, 17: 15, 9: 16, 10: 17}
        else:
            raise NotImplementedError("Unavailable number of classes")

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s_gtFine_labelids.png" % (self.set, name[:-16]))
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
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

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
    dst = IDDDataSet('working_directory/data/IDD', './idd_list/train.txt',
                     crop_size=(512, 256), ignore_label=255, set='train', num_classes=18)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, name = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()

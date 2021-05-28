import argparse
import numpy as np
import random

import torch
from torch.utils import data
from model.deeplab_multi import Deeplab_multi
from dataset.cityscapes_dataset import CityScapesDataSet
from dataset.idd_dataset import IDDDataSet
import os
from PIL import Image
import torch.nn as nn

NUM_TARGET = 2
EVAL_TARGET = -1
TM = False
DIR_NAME = ''

CityScapes = False
IDD = True

BATCH_SIZE = 1

DATA_DIRECTORY_CityScapes = './data/CityScapes'
DATA_LIST_PATH_CityScapes = './dataset/cityscapes_list/val.txt'

DATA_DIRECTORY_IDD = './data/IDD'
DATA_LIST_PATH_IDD = './dataset/idd_list/val.txt'

IGNORE_LABEL = 255

NUM_CLASSES = 18

SET = 'val'

RANDOM_SEED = 1338

SAVE = './plots'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ETM framework")
    parser.add_argument("--num-target", type=int, default=NUM_TARGET)
    parser.add_argument("--eval-target", type=int, default=EVAL_TARGET
                        , help="choose which TM to use")
    parser.add_argument("--cityscapes", action='store_true', default=CityScapes)
    parser.add_argument("--idd", action='store_true', default=IDD)
    parser.add_argument("--data-dir-cityscapes", type=str, default=DATA_DIRECTORY_CityScapes)
    parser.add_argument("--data-list-cityscapes", type=str, default=DATA_LIST_PATH_CityScapes)
    parser.add_argument("--data-dir-idd", type=str, default=DATA_DIRECTORY_IDD)
    parser.add_argument("--data-list-idd", type=str, default=DATA_LIST_PATH_IDD)
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                             help="Number of images sent to the network in one step.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--dir-name", type=str, default=DIR_NAME)
    parser.add_argument("--save", type=str, default=SAVE)
    parser.add_argument("--tm", action='store_true', default=TM)
    return parser.parse_args()

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (1024, 512)

    model = Deeplab_multi(args=args)

    saved_state_dict = torch.load('./snapshots/' + args.dir_name)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        if i in new_params.keys():
            new_params[i] = saved_state_dict[i]
    model.load_state_dict(new_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    interp_cityscapes = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    interp_idd = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    if args.cityscapes:
        if not os.path.exists(os.path.join(args.save, args.dir_name.split('/')[0], 'CityScapes')):
            os.makedirs(os.path.join(args.save, args.dir_name.split('/')[0], 'CityScapes'))

        cityscapes_loader = torch.utils.data.DataLoader(
            CityScapesDataSet(args.data_dir_cityscapes, args.data_list_cityscapes,
                              crop_size=input_size, ignore_label=args.ignore_label,
                              set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        for index, batch in enumerate(cityscapes_loader):
            if index % 100 == 0:
                print('%d processd (CityScapes)' % index)
            image, _, name = batch
            image = image.to(device)

            if args.tm:
                pred, _, _, _ = model(image, input_size)
            else:
                _, _, pred, _ = model(image, input_size)
            output = interp_cityscapes(pred).cpu().data[0].numpy()

            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)

            name = name[0].split('/')[-1]
            output_col.save('%s/%s/CityScapes/%s_color.png' % (args.save, args.dir_name.split('/')[0],
                                                               name.split('.')[0]))

    if args.idd:
        if not os.path.exists(os.path.join(args.save, args.dir_name.split('/')[0], 'IDD')):
            os.makedirs(os.path.join(args.save, args.dir_name.split('/')[0], 'IDD'))

        idd_loader = torch.utils.data.DataLoader(
            IDDDataSet(args.data_dir_idd, args.data_list_idd,
                       crop_size=input_size, ignore_label=args.ignore_label,
                       set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        for index, batch in enumerate(idd_loader):
            if index % 100 == 0:
                print('%d processd (IDD)' % index)
            image, _, name = batch
            image = image.to(device)

            if args.tm:
                pred, _, _, _ = model(image, input_size)
            else:
                _, _, pred, _ = model(image, input_size)
            output = interp_idd(pred).cpu().data[0].numpy()

            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)

            name = name[0].split('/')[-1]
            output_col.save('%s/%s/IDD/%s_color.png' % (args.save, args.dir_name.split('/')[0],
                                                        name.split('.')[0]))


if __name__ == '__main__':
    main()

import argparse
import numpy as np
import random

import torch
from torch.utils import data
from model.deeplab_multi import Deeplab_multi
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.crosscity_dataset import CrossCityDataSet

SOURCE = 'GTA5'  # 'GTA5' or 'SYNTHIA'
NUM_TARGET = 1
EVAL_TARGET = -1
TM = False
DIR_NAME = ''

GTA5 = False
SYNTHIA = False
RIO = False
ROME = False
TAIPEI = False
TOKYO = False
PER_CLASS = True

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000

BATCH_SIZE = 1

DATA_DIRECTORY_GTA5 = './data/GTA5'
DATA_LIST_PATH_GTA5 = './dataset/gta5_list/val.txt'

DATA_DIRECTORY_SYNTHIA = './data/SYNTHIA'
DATA_LIST_PATH_SYNTHIA = './dataset/synthia_list/val.txt'

DATA_DIRECTORY_TARGET = './data/NTHU_Datasets'

IGNORE_LABEL = 255
NUM_CLASSES = 13

SET = 'val'
SET_TARGET = 'test'

RANDOM_SEED = 1338

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CUDA square framework")
    parser.add_argument("--source", action='store_true', default=SOURCE)
    parser.add_argument("--num-target", type=int, default=NUM_TARGET)
    parser.add_argument("--eval-target", type=int, default=EVAL_TARGET
                        , help="choose which TM to use")
    parser.add_argument("--gta5", action='store_true', default=GTA5)
    parser.add_argument("--synthia", action='store_true', default=SYNTHIA)
    parser.add_argument("--rio", action='store_true', default=RIO)
    parser.add_argument("--rome", action='store_true', default=ROME)
    parser.add_argument("--taipei", action='store_true', default=TAIPEI)
    parser.add_argument("--tokyo", action='store_true', default=TOKYO)
    parser.add_argument("--mIoUs-per-class", action='store_true', default=PER_CLASS)
    parser.add_argument("--data-dir-gta5", type=str, default=DATA_DIRECTORY_GTA5)
    parser.add_argument("--data-list-gta5", type=str, default=DATA_LIST_PATH_GTA5)
    parser.add_argument("--data-dir-synthia", type=str, default=DATA_DIRECTORY_SYNTHIA)
    parser.add_argument("--data-list-synthia", type=str, default=DATA_LIST_PATH_SYNTHIA)
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET)
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                             help="Number of images sent to the network in one step.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--set_target", type=str, default=SET_TARGET,
                        help="choose evaluation set.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--dir-name", type=str, default=DIR_NAME)
    parser.add_argument("--tm", action='store_true', default=TM)
    return parser.parse_args()


def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (1024, 512)

    name_classes = np.asarray(["road",
                               "sidewalk",
                               "building",
                               "light",
                               "sign",
                               "vegetation",
                               "sky",
                               "person",
                               "rider",
                               "car",
                               "bus",
                               "motorcycle",
                               "bicycle"])

    # Create the model and start the evaluation process
    model = Deeplab_multi(args=args)
    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        saved_state_dict = torch.load('./snapshots/' + args.dir_name + '/' + str((files + 1) * args.save_pred_every) + '.pth')
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.eval()
        if args.gta5:
            gta5_loader = torch.utils.data.DataLoader(
                GTA5DataSet(args.data_dir_gta5, args.data_list_gta5,
                            crop_size=input_size, ignore_label=args.ignore_label,
                            set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(gta5_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.tm:
                    pred, _, _, _ = model(images_val, input_size)
                else:
                    _, _, pred, _ = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (GTA5): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if args.synthia:
            synthia_loader = torch.utils.data.DataLoader(
                SYNTHIADataSet(args.data_dir_synthia, args.data_list_synthia,
                               crop_size=input_size, ignore_label=args.ignore_label,
                               set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(synthia_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.tm:
                    pred, _, _, _ = model(images_val, input_size)
                else:
                    _, _, pred, _ = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (SYNTHIA): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if args.rio:
            crosscity_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, 'Rio',
                                 crop_size=input_size, ignore_label=args.ignore_label,
                                 set=args.set_target, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(crosscity_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.tm:
                    pred, _, _, _ = model(images_val, input_size)
                else:
                    _, _, pred, _ = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Rio): {}'.format(str(round(np.nanmean(mIoUs) * 100, 2))))
            print('=' * 50)

        if args.rome:
            crosscity_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, 'Rome',
                                 crop_size=input_size, ignore_label=args.ignore_label,
                                 set=args.set_target, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(crosscity_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.tm:
                    pred, _, _, _ = model(images_val, input_size)
                else:
                    _, _, pred, _ = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Rome): {}'.format(str(round(np.nanmean(mIoUs) * 100, 2))))
            print('=' * 50)

        if args.taipei:
            crosscity_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, 'Taipei',
                                 crop_size=input_size, ignore_label=args.ignore_label,
                                 set=args.set_target, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(crosscity_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.tm:
                    pred, _, _, _ = model(images_val, input_size)
                else:
                    _, _, pred, _ = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (taipei): {}'.format(str(round(np.nanmean(mIoUs) * 100, 2))))
            print('=' * 50)

        if args.tokyo:
            crosscity_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, 'Tokyo',
                                 crop_size=input_size, ignore_label=args.ignore_label,
                                 set=args.set_target, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(crosscity_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.tm:
                    pred, _, _, _ = model(images_val, input_size)
                else:
                    _, _, pred, _ = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Tokyo): {}'.format(str(round(np.nanmean(mIoUs) * 100, 2))))
            print('=' * 50)

if __name__ == '__main__':
    main()

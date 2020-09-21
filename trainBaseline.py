import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import random
from options_Baseline import TrainOptions
from model.deeplab_multi import Deeplab_multi
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet

args = TrainOptions().parse()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if args.from_scratch:
        optimizer.param_groups[1]['lr'] = lr
    else:
        optimizer.param_groups[1]['lr'] = lr * 10
    if args.tm:
        optimizer.param_groups[2]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr


def main():
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    cudnn.enabled = True

    # Create network
    model = Deeplab_multi(args=args)
    saved_state_dict = torch.load(args.restore_from_resnet, map_location=device)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    model.train()
    model.to(device)

    cudnn.benchmark = True

    # Dataloader
    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                        crop_size=input_size, ignore_label=args.ignore_label,
                        set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    elif args.source == 'SYNTHIA':
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                           crop_size=input_size, ignore_label=args.ignore_label,
                           set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError('Unavailable source domain')
    trainloader_iter = enumerate(trainloader)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # start training
    for i_iter in range(args.num_steps):

        loss_seg_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        _, batch = trainloader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        _, _, pred, _ = model(images, input_size)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value += loss_seg.item()

        loss.backward()
        optimizer.step()

        print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
        print('iter = {0:8d}/{1:8d}'.format(i_iter, args.num_steps))
        print('loss_seg = {0:.3f}'.format(loss_seg_value))

        # Snapshots directory
        if not os.path.exists(osp.join(args.snapshot_dir, args.dir_name)):
            os.makedirs(osp.join(args.snapshot_dir, args.dir_name))

        # Save model
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '.pth'))

if __name__ == '__main__':
    main()

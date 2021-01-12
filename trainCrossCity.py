import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random
import copy
from options_trainCrossCity import TrainOptions
from model.deeplab_multi import Deeplab_multi
from model.discriminator import FCDiscriminator, DHA
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.crosscity_dataset import CrossCityDataSet

PRE_TRAINED_SEG = './snapshots/*/*.pth'

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

def distillation_loss(pred_origin, old_outputs):
    pred_origin_logsoftmax = (pred_origin / 2).log_softmax(dim=1)
    old_outputs = (old_outputs / 2).softmax(dim=1)
    loss_distillation = (-(old_outputs * pred_origin_logsoftmax)).sum(dim=1)
    loss_distillation = loss_distillation.sum() / loss_distillation.flatten().shape[0]
    return loss_distillation

def prob_2_entropy(prob):
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

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
    if args.from_scratch:  # training model from pre-trained ResNet
        saved_state_dict = torch.load(args.restore_from_resnet, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:  # training model from pre-trained DeepLabV2 on source & previous target domains
        saved_state_dict = torch.load(PRE_TRAINED_SEG, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)
    model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)

    model_D1.train()
    model_D1.to(device)

    model_D2.train()
    model_D2.to(device)

    # reference model
    if not args.from_scratch:
        ref_model = copy.deepcopy(model)  # reference model for knowledge distillation
        for params in ref_model.parameters():
            params.requires_grad = False
        ref_model.eval()

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

    targetloader = data.DataLoader(CrossCityDataSet(args.data_dir_target, args.target,
                                                    max_iters=args.num_steps * args.batch_size,
                                                    crop_size=input_size,
                                                    ignore_label=args.ignore_label,
                                                    set=args.set, num_classes=args.num_classes),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'DHA':
        adversarial_loss_1 = DHA(model_D1)
        adversarial_loss_2 = DHA(model_D2)
    else:
        raise NotImplementedError('Unavailable GAN option')

    # labels for adversarial training
    source_label = 1
    target_label = 0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # start training
    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_value1 = 0
        loss_distill_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_value2 = 0
        loss_distill_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        # train f

        # freeze D
        for param in model_D1.parameters():
            param.requires_grad = False

        for param in model_D2.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        if args.tm:
            pred2, pred1, pred_ori2, pred_ori1 = model(images, input_size)
        else:
            _, _, pred2, pred1 = model(images, input_size)

        loss_seg1 = seg_loss(pred1, labels)
        loss_seg2 = seg_loss(pred2, labels)
        loss = args.lambda_seg1 * loss_seg1 + args.lambda_seg2 * loss_seg2
        loss_seg_value1 += loss_seg1.item()
        loss_seg_value2 += loss_seg2.item()

        if not args.from_scratch and args.tm:
            _, _, old_outputs2, old_outputs1 = ref_model(images, input_size)
            loss_distill1 = distillation_loss(pred_ori1, old_outputs1)
            loss_distill2 = distillation_loss(pred_ori2, old_outputs2)
            loss += args.lambda_distill1 * loss_distill1 + args.lambda_distill2 * loss_distill2
            loss_distill_value1 += loss_distill1.item()
            loss_distill_value2 += loss_distill2.item()

        if not args.gan == 'DHA':
            loss.backward()

        _, batch = targetloader_iter.__next__()
        images_target, _, _ = batch
        images_target = images_target.to(device)

        if args.tm:
            pred_target2, pred_target1, _, _ = model(images_target, input_size)
        else:
            _, _, pred_target2, pred_target1 = model(images_target, input_size)

        if args.gan == 'DHA':
            if args.ent:
                loss_adv1 = adversarial_loss_1(prob_2_entropy(F.softmax(pred_target1, dim=1)),
                                               prob_2_entropy(F.softmax(pred1, dim=1)),
                                               loss_type='adversarial')
                loss_adv2 = adversarial_loss_2(prob_2_entropy(F.softmax(pred_target2, dim=1)),
                                               prob_2_entropy(F.softmax(pred2, dim=1)),
                                               loss_type='adversarial')
            else:
                loss_adv1 = adversarial_loss_1(F.softmax(pred_target1, dim=1),
                                               F.softmax(pred1, dim=1),
                                               loss_type='adversarial')
                loss_adv2 = adversarial_loss_2(F.softmax(pred_target2, dim=1),
                                               F.softmax(pred2, dim=1),
                                               loss_type='adversarial')

            loss += args.lambda_adv1 * loss_adv1 + args.lambda_adv2 * loss_adv2
        elif args.gan == 'Vanilla':
            if args.ent:
                D_out1 = model_D1(prob_2_entropy(F.softmax(pred_target1, dim=1)))
                D_out2 = model_D2(prob_2_entropy(F.softmax(pred_target2, dim=1)))
            else:
                D_out1 = model_D1(F.softmax(pred_target1, dim=1))
                D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_adv1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_adv2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv1 * loss_adv1 + args.lambda_adv2 * loss_adv2
        else:
            raise NotImplementedError('Unavailable GAN option')

        loss_adv_value1 += loss_adv1.item()
        loss_adv_value2 += loss_adv2.item()
        loss.backward()

        # train D

        # bring back requires_grad
        for param in model_D1.parameters():
            param.requires_grad = True

        for param in model_D2.parameters():
            param.requires_grad = True

        pred1 = pred1.detach()
        pred2 = pred2.detach()
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        if args.gan == 'DHA':
            if args.ent:
                loss_D1 = adversarial_loss_1(prob_2_entropy(F.softmax(pred_target1, dim=1)),
                                               prob_2_entropy(F.softmax(pred1, dim=1)),
                                               loss_type='discriminator')
                loss_D2 = adversarial_loss_2(prob_2_entropy(F.softmax(pred_target2, dim=1)),
                                               prob_2_entropy(F.softmax(pred2, dim=1)),
                                               loss_type='discriminator')
            else:
                loss_D1 = adversarial_loss_1(F.softmax(pred_target1, dim=1),
                                               F.softmax(pred1, dim=1),
                                               loss_type='discriminator')
                loss_D2 = adversarial_loss_2(F.softmax(pred_target2, dim=1),
                                               F.softmax(pred2, dim=1),
                                               loss_type='discriminator')

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
        elif args.gan == 'Vanilla':
            # train with source
            if args.ent:
                D_out1 = model_D1(prob_2_entropy(F.softmax(pred1, dim=1)))
                D_out2 = model_D2(prob_2_entropy(F.softmax(pred2, dim=1)))
            else:
                D_out1 = model_D1(F.softmax(pred1, dim=1))
                D_out2 = model_D2(F.softmax(pred2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

            # train with target
            if args.ent:
                D_out1 = model_D1(prob_2_entropy(F.softmax(pred_target1, dim=1)))
                D_out2 = model_D2(prob_2_entropy(F.softmax(pred_target2, dim=1)))
            else:
                D_out1 = model_D1(F.softmax(pred_target1, dim=1))
                D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
        else:
            raise NotImplementedError('Unavailable GAN option')

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
        print('iter = {0:8d}/{1:8d}'.format(i_iter, args.num_steps))
        print('loss_seg1 = {0:.3f} loss_dist1 = {1:.3f} loss_adv1 = {2:.3f} loss_D1 = {3:.3f}'.format(
            loss_seg_value1, loss_distill_value1, loss_adv_value1, loss_D_value1))
        print('loss_seg2 = {0:.3f} loss_dist2 = {1:.3f} loss_adv2 = {2:.3f} loss_D2 = {3:.3f}'.format(
            loss_seg_value2, loss_distill_value2, loss_adv_value2, loss_D_value2))

        # Snapshots directory
        if not os.path.exists(osp.join(args.snapshot_dir, args.dir_name)):
            os.makedirs(osp.join(args.snapshot_dir, args.dir_name))

        # Save model
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(),
                       osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(),
                       osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '_D2.pth'))

if __name__ == '__main__':
    main()

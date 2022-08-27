import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import cos, pi
from copy import deepcopy
import sys
import torch.nn.functional as F
from image_classification.compute_flops import print_model_param_flops
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from . import logger as log
import time
from torch.distributions import MultivariateNormal
from . import utils


class DI(nn.Module):
    def __init__(self, channels):
        super(DI, self).__init__()
        self.mask = nn.Parameter(torch.ones(channels), requires_grad=True)

    def forward(self, X, Y):
        # X: BxCxHxW
        # Y: B,
        B, C, H, W = X.shape
        X_ = (X * self.mask.view(1, C, 1, 1)).view(B, C*H*W)
        X_ = X_ - torch.mean(X_, dim=0, keepdim=True)
        Y_ = F.one_hot(Y, 1000).float() # imagenet has 1000 classes
        Y_ = Y_ - torch.mean(Y_, dim=0, keepdim=True)
        outer_Y = torch.matmul(Y_, Y_.t())
        outer_X = torch.matmul(X_, X_.t()) + 0.1 * torch.eye(B).to(X.get_device())
        return torch.matmul(outer_Y, torch.inverse(outer_X))


def get_rank(input_model, data_loader):
    model = deepcopy(input_model)
    model.eval()

    list_conv = []
    def conv_hook(self, input, output):
        di = DI(output.shape[1]).cuda()
        di.train()
        loss = torch.trace(di(output.detach(), target.detach()))
        loss.backward()
        grad = di.mask.grad.data.abs().detach().cpu().numpy()
        list_conv.append(grad)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            return
        for c in childrens:
            foo(c)
    foo(model)

    list_list = []
    for idx, (data, target) in enumerate (data_loader):
        print(idx)
        if idx >= 10:
            break
        data, target = Variable(data.cuda()), Variable(target.cuda())
        model(data)
        if idx == 0:
            score = list_conv
        else:
            score = [(np.array(x)+np.array(y)).tolist() for x, y in zip(score, list_conv)]
        list_conv = []
    full_rank = [np.argsort(m) for m in score]

    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    rank = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if layer_id in l1 + l2 + skip:
                rank.append(full_rank[layer_id-1])
                layer_id += 1
                continue
            layer_id += 1
    return rank


def L1_norm(layer):
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    norm = np.sum(weight_copy, axis=(1,2,3))
    return norm


def get_channel_mask(model, ratio, rank):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    idx = 0
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1 + l2 + skip:
                num_keep = int(out_channels * (1 - ratio[idx]))
                # rank = np.argsort(L1_norm(m))
                arg_max_rev = rank[idx][::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                idx += 1
                continue
            layer_id += 1
    return cfg_mask


def apply_channel_mask(model, cfg_mask):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id_in_cfg = 0
    conv_count = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if conv_count in l1:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l2:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                prev_mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1)
                m.weight.data.mul_(prev_mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l3:
                prev_mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1)
                m.weight.data.mul_(prev_mask)
                conv_count += 1
                continue
            if conv_count in skip:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            conv_count += 1
        elif isinstance(m, nn.BatchNorm2d):
            if conv_count-1 in l1 + l2 + skip:
                mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue


def check_layer_channel_sparsity (model):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    res = []
    conv_count = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if conv_count in l1 + l2 + skip:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                norm = np.sum(weight_copy, axis=(1,2,3))
                num_zeros = len(np.where(norm == 0)[0])
                res.append(num_zeros / m.weight.data.shape[0])
                conv_count += 1
                continue
            conv_count += 1
    return res


def flops_to_ratio (input_model, max_flops, p, current_ratio, rank):
    current_flops = print_model_param_flops(input_model, 224, False)
    reduction = []
    for i in range(len(current_ratio)):
        tmp_ratio = current_ratio.copy()
        if tmp_ratio[i] >= 0.9:
            reduction.append(None)
        else:
            tmp_ratio[i] = min(1, tmp_ratio[i]+0.1)
            model = deepcopy(input_model)
            cfg_mask = get_channel_mask(model, tmp_ratio, rank)
            apply_channel_mask(model, cfg_mask)
            tmp_flops = print_model_param_flops(model, 224, False)
            reduction.append(current_flops - tmp_flops)
            assert (current_flops - tmp_flops) > 0
    max_reduction = max_flops * p
    res = []
    for i in range(len(current_ratio)):
        if reduction[i] is None:
            res.append(None)
        else:
            res.append(0.1 * max_reduction / reduction[i])
    return res


def test(model, val_loader):
    model.eval()
    top1 = log.AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            bs = input.size(0)
            input_var = Variable(input)
            target_var = Variable(target)
            output = model(input_var)
            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
            torch.cuda.synchronize()
            top1.record(prec1.item(), bs)
    return top1.get_val()[0]


def ratio_to_acc (input_model, layer_reduction, train_loader, val_loader, current_ratio, rank):
    res = []
    for i in range(len(current_ratio)):
        if layer_reduction[i] is None:
            res.append(-1)
        else:
            model = deepcopy(input_model)
            prune_ratio = current_ratio.copy()
            prune_ratio[i] = min(1, prune_ratio[i] + layer_reduction[i])
            cfg_mask = get_channel_mask(model, prune_ratio, rank)
            apply_channel_mask(model, cfg_mask)
            res.append(test(model, val_loader))
    return res


def iterative_prune (model, train_loader, val_loader, N, max_flops, p):
    print('Iteratively prune', N, 'steps, then finetune 1 epoch')
    for idx in range(N):

        rank = get_rank(model, val_loader) # intra-layer channel ranking

        current_ratio = check_layer_channel_sparsity(model)
        print('Pruning iteration:', idx, 'PreRatio:', current_ratio)

        layer_reduction = flops_to_ratio(model, max_flops, p, current_ratio, rank)
        print('Pruning iteration:', idx, 'Candidate:', layer_reduction)

        layer_acc = ratio_to_acc(model, layer_reduction, train_loader, val_loader, current_ratio, rank)
        print('Pruning iteration:', idx, 'Accuracy:', layer_acc)

        winner_idx = layer_acc.index(max(layer_acc))
        prune_ratio = current_ratio.copy()
        prune_ratio[winner_idx] = min(1, prune_ratio[winner_idx] + layer_reduction[winner_idx])
        print('Pruning iteration:', idx, 'Winner', winner_idx, 'PostRatio:', prune_ratio)

        cfg_mask = get_channel_mask(model, prune_ratio, rank)
        apply_channel_mask(model, cfg_mask)
        print('Pruning iteration:', idx, 'FLOPs:', print_model_param_flops(model, 224, False))

    return cfg_mask



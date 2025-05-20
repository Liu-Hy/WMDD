'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import torch
import torch.nn as nn
from torch import distributed
import numpy as np
import json
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_arguments(script_name, args):
    directory = "../log/"
    filename = os.path.join(directory, f"{args.exp_name}.json")
    os.makedirs(directory, exist_ok=True)

    # If the JSON file exists, load its data. If not, initialize an empty list.
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []

    found = False
    for entry in data:
        if entry["script"] == script_name:
            # Update the arguments for this script
            entry["args"] = vars(args)
            found = True
            break

    # If script_name was not found in the list, append a new entry
    if not found:
        entry = {
            "script": script_name,
            "args": vars(args)
        }
        data.append(entry)

    # Save the updated data back to the JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


class BNFeatureHook_new():
    """For the new version of per-class batchnorm"""
    def __init__(self, module, per_class_bn=False, ipc=None, start=None, end=None, weights=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.per_class_bn = per_class_bn
        self.ipc = ipc
        self.start = start
        self.end = end
        # Convert weights to tensor if they're not already
        if weights is not None and not isinstance(weights, torch.Tensor):
            self.weights = torch.tensor(weights)
        else:
            self.weights = weights

    def hook_fn(self, module, input, output):
        if self.per_class_bn:
            num_cls = self.end - self.start
            _, nch, h, w = input[0].shape
            
            if self.weights is not None:
                # Get the weights for the current classes
                class_weights = self.weights[self.start:self.end]
                
                # Reshape weights for broadcasting: [num_cls, ipc, 1, 1, 1]
                weights_reshaped = class_weights.view(num_cls, self.ipc, 1, 1, 1).to(input[0].device)
                
                # Normalize weights across samples for each class
                weights_normalized = weights_reshaped / weights_reshaped.sum(dim=1, keepdim=True)
                
                # Reshape input: [batch, channel, h, w] -> [num_cls, ipc, channel, h, w]
                inp = input[0].view(num_cls, self.ipc, nch, h, w)
                
                # Weighted mean: sum(w_i * x_i) / sum(w_i)
                # First compute the weighted sum over 'ipc' dimension (dim=1)
                # Then compute mean over spatial dimensions (h, w) which are now dim=2,3
                mean = (inp * weights_normalized).sum(dim=1).mean(dim=[2, 3])
                
                # Weighted variance calculation
                # First center each input by the class mean
                centered = inp - mean.view(num_cls, 1, nch, 1, 1)
                # Then calculate weighted variance
                var = (weights_normalized * centered**2).sum(dim=1).mean(dim=[2, 3])
            else:
                # Original unweighted calculation
                inp = input[0].view(num_cls, self.ipc, nch, h, w)
                mean = inp.mean([1, 3, 4])  # Average over ipc, h, w dimensions
                var = inp.var([1, 3, 4], unbiased=False)

            diff_var = module.running_var_per_class.data[self.start: self.end] - var
            diff_mean = module.running_mean_per_class.data[self.start: self.end] - mean

            r_feature = torch.norm(diff_var, dim=1).mean() + torch.norm(diff_mean, dim=1).mean()
            #r_feature = (torch.sum(diff_var ** 2, dim=1).mean()) + (torch.sum(diff_mean ** 2, dim=1).mean())
            self.r_feature = r_feature

        else:
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
                module.running_mean.data - mean, 2)
            self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class BNFeatureHook():
    def __init__(self, module, per_class_bn=False, start=None, end=None, weights=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.per_class_bn = per_class_bn
        self.start = start
        self.end = end
        # Convert weights to tensor if they're not already
        if weights is not None and not isinstance(weights, torch.Tensor):
            self.weights = torch.tensor(weights)
        else:
            self.weights = weights

    def hook_fn(self, module, input, output):
        if self.per_class_bn:
            if self.weights is not None:
                # Get weights for current classes
                class_weights = self.weights[self.start:self.end]
                batch_size = input[0].shape[0]
                
                # Convert weights to torch tensor if it's not already
                if not isinstance(class_weights, torch.Tensor):
                    class_weights = torch.tensor(class_weights)
                
                # Reshape weights to match batch dimensions
                # For older implementation, the batch dimension contains all class samples together
                weights_tensor = class_weights.flatten().to(input[0].device)
                
                # Ensure we have the right number of weights
                if weights_tensor.shape[0] != batch_size:
                    # If not matching, default to equal weights
                    weights_tensor = torch.ones(batch_size, device=input[0].device)
                
                # Normalize weights to sum to 1
                weights_normalized = weights_tensor / weights_tensor.sum()
                weights_for_batch = weights_normalized.view(batch_size, 1, 1, 1)
                
                # Calculate spatial means for each sample
                spatial_mean = input[0].mean([2, 3])  # [batch, channels]
                
                # Apply weights to compute weighted mean across batch dimension
                mean = (spatial_mean * weights_normalized.view(batch_size, 1)).sum(dim=0)
                
                # For variance, first compute per-sample spatial variance
                centered = input[0] - spatial_mean.view(batch_size, -1, 1, 1)
                spatial_var = (centered**2).mean([2, 3])  # [batch, channels]
                
                # Apply weights to compute weighted variance
                var = (spatial_var * weights_normalized.view(batch_size, 1)).sum(dim=0)
            else:
                # Original unweighted calculation
                mean = input[0].mean([2, 3])
                var = input[0].var([2, 3], unbiased=False)

            diff_var = module.running_var_per_class.data[self.start: self.end] - var
            diff_mean = module.running_mean_per_class.data[self.start: self.end] - mean

            r_feature = torch.norm(diff_var, dim=1).mean() + torch.norm(diff_mean, dim=1).mean()
            self.r_feature = r_feature

        else:
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
                module.running_mean.data - mean, 2)
            self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class EmbedderHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output.view(output.shape[0], -1)

    def close(self):
        self.hook.remove()


class BatchNorm2dWithPerClassStats(nn.BatchNorm2d):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super(BatchNorm2dWithPerClassStats, self).__init__(num_features, *args, **kwargs)

        self.num_classes = num_classes
        self.register_buffer('accumulated_mean_per_class', torch.zeros(num_classes, num_features))
        self.register_buffer('accumulated_var_per_class', torch.zeros(num_classes, num_features))
        self.register_buffer('samples_seen_per_class', torch.zeros(num_classes, dtype=torch.long))

    def forward(self, input, labels=None):
        if labels is not None and not self.training:
            if torch.isnan(input).any() or torch.isinf(input).any():
                raise ValueError("NaN or Inf values detected in the input.")
            assert labels.dim() < 2, print(labels.shape)
            c = labels.item()

            with torch.no_grad():
                # Compute the batch mean and variance
                class_mean = input.mean(dim=[0, 2, 3])
                class_var = input.var(dim=[0, 2, 3], unbiased=False)

                # Update accumulated mean and variance
                self.accumulated_mean_per_class[c] += class_mean * input.size(0)  # weight by number of samples
                self.accumulated_var_per_class[c] += class_var * input.size(0)
                self.samples_seen_per_class[c] += input.size(0)
        return super(BatchNorm2dWithPerClassStats, self).forward(input)

    def finalize_statistics(self):
        with torch.no_grad():
            self.running_mean_per_class = self.accumulated_mean_per_class / self.samples_seen_per_class.unsqueeze(1)
            self.running_var_per_class = self.accumulated_var_per_class / self.samples_seen_per_class.unsqueeze(1)


def replace_batchnorm(module, num_classes):
    """
    Recursively replaces all batch norm layers in the module and its sub-modules.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, BatchNorm2dWithPerClassStats(child.num_features, num_classes))
        else:
            replace_batchnorm(child, num_classes)

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2
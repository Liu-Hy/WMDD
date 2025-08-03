import torch
import numpy as np
import json
import os


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

# keep top k largest values, and smooth others
def keep_top_k(p,k,n_classes=1000): # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes-k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(topk_smooth_p.sum().item(), p.shape[0]), f'{topk_smooth_p.sum().item()} not close to {p.shape[0]}'
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups





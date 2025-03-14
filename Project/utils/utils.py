import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset), collate_fn=collate_MIL, **kwargs)
    return loader 

def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
    Returns either the validation loader or training loader.
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)), collate_fn=collate_MIL, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset), collate_fn=collate_MIL, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset), collate_fn=collate_MIL, **kwargs)
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate_MIL, **kwargs)

    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None, mandatory_train_ids=None):
    # Generate a complete list of sample indices
    indices = np.arange(samples).astype(int)
    
    # If a custom test set is provided, remove those samples from all samples
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)
    
    # If mandatory training samples exist, exclude them from other splits
    if mandatory_train_ids is not None:
        indices = np.setdiff1d(indices, mandatory_train_ids)
    ori_indices = indices
    np.random.seed(11)

    for i in range(n_splits):
        indices = ori_indices
        # Initialize train, validation, and test sets
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # If custom test set exists, do not sample for test set
            all_test_ids.extend(custom_test_ids)

        # Randomly sample validation set
        val_ids = np.random.choice(indices, sum(val_num), replace=False)
        indices = np.setdiff1d(indices, val_ids)  # Remove validation set samples
        all_val_ids.extend(val_ids)

        # Randomly sample test set
        test_ids = np.random.choice(indices, sum(test_num), replace=False)
        indices = np.setdiff1d(indices, test_ids)  # Remove test set samples
        all_test_ids.extend(test_ids)

        # Use remaining samples as the training set, sample according to label fraction
        if label_frac == 1.0:
            sampled_train_ids.extend(indices)
        else:
            sample_num = math.ceil(len(indices) * label_frac)
            sampled_train_ids.extend(indices[:sample_num])

        if mandatory_train_ids is not None:
            mandatory_train_ids = np.array(mandatory_train_ids)
            np.random.shuffle(mandatory_train_ids)  # Shuffle mandatory training samples

            # Move 6 to validation
            moved_to_val = mandatory_train_ids[:6]
            all_val_ids.extend(moved_to_val)

            # Keep the rest in training
            remaining_mandatory_train = mandatory_train_ids[6:]
            sampled_train_ids.extend(remaining_mandatory_train)

            if len(all_val_ids) >= 6:
                moved_to_train = np.random.choice(all_val_ids, 6, replace=False)
                all_val_ids = np.setdiff1d(all_val_ids, moved_to_train)  # Remove from validation
                sampled_train_ids.extend(moved_to_train)  # Add to training
        yield sampled_train_ids, all_val_ids, all_test_ids

def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

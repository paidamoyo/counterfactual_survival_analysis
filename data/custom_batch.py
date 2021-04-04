# https://pytorch.org/docs/stable/data.html
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.x = torch.stack(transposed_data[0], 0)
        self.y = torch.stack(transposed_data[1], 0)
        self.e = torch.stack(transposed_data[2], 0)
        self.a = torch.stack(transposed_data[3], 0)
        self.c_w = torch.stack(transposed_data[4], 0)
        self.t_w = torch.stack(transposed_data[5], 0)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        self.e = self.e.pin_memory()
        self.a = self.a.pin_memory()
        self.c_w = self.c_w.pin_memory()
        self.t_w = self.t_w.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def build_iterator(args, train_data, test_data, valid_data):
    train_iterator = DataLoader(
        convert_tensor(x=train_data['x'], y=train_data['y'], e=train_data['e'], a=train_data['a'],
                       c_w=train_data['c_w'], t_w=train_data['t_w']),
        batch_size=args.batch_size, shuffle=True, sampler=None, batch_sampler=None,
        collate_fn=collate_wrapper,
        pin_memory=True, num_workers=1, drop_last=True, worker_init_fn=worker_init_fn(args=args))

    valid_iterator = DataLoader(
        convert_tensor(x=valid_data['x'], y=valid_data['y'], e=valid_data['e'], a=valid_data['a'],
                       c_w=valid_data['c_w'], t_w=valid_data['t_w']),
        batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
        collate_fn=collate_wrapper,
        pin_memory=True, num_workers=1, worker_init_fn=worker_init_fn(args=args))

    test_iterator = DataLoader(
        convert_tensor(x=test_data['x'], y=test_data['y'], e=test_data['e'], a=test_data['a'],
                       c_w=test_data['c_w'], t_w=test_data['t_w']),
        batch_size=args.batch_size, shuffle=False, sampler=None, batch_sampler=None,
        collate_fn=collate_wrapper,
        pin_memory=True, num_workers=1, worker_init_fn=worker_init_fn(args=args))

    return {"train_iterator": train_iterator, "valid_iterator": valid_iterator, "test_iterator": test_iterator}


def convert_tensor(x, y, e, a, c_w, t_w):
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
                         torch.from_numpy(e), torch.from_numpy(a), torch.from_numpy(c_w), torch.from_numpy(t_w))


def worker_init_fn(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

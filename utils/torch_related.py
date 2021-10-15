import torch
from torch.utils.data import Dataset
import numpy as np
import random


def setup_seed(seed = 42):
    """the answer to world and life.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def dict_to_list_by_max_len(batch):
    """按每个batch中句子的最大长度截取输入向量

    batch: [{"length": int, ...}, ...] -> len(batch) = batch_size
    """
    keys = list(batch[0].keys())
    batch_values = [list(i.values()) for i in batch]
    batch_values = map(torch.stack, zip(*batch_values))
    batch_dict = dict(zip(keys, batch_values))
    max_len = int(max(batch_dict["length"]).item())
    for i in batch_dict:
        if i != "length":
            batch_dict[i] = batch_dict[i][:, :max_len]
    return batch_dict


class MyDataSet(Dataset):
    """传入的数据为字典类型
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return self.kwargs[list(self.kwargs.keys())[0]].size()[0]

    def __getitem__(self, id):
        return {i: self.kwargs[i][id] for i in self.kwargs}
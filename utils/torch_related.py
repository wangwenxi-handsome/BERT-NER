import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_torch_model(
    model_cls,
    model_config = {},
    load_checkpoint_path = None,
):
    """自动识别设备（默认使用全部gpu）并调整到相应模式. 
    支持的模式有cpu模型，单机单卡模式，单机多卡模式
    !!!TODO support DDP mode
    """
    """
    if torch.cuda.device_count() <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if_DDP_mode = False
    else:
        # DDP：从外部得到local_rank参数
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", default=-1, type=int)
        FLAGS = parser.parse_args()
        device = FLAGS.local_rank
        if_DDP_mode = True

        # DDP：DDP backend初始化
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    # load model
    if load_checkpoint_path is not None:
        model = torch.load(load_checkpoint_path).to(device)
    else:
        model = model_cls(**model_config).to(device)
    
    if if_DDP_mode:
        # DDP: 构造DDP model
        model = DDP(model, device_ids=[device], output_device=device)
    return device, model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if load_checkpoint_path is not None:
        model = torch.load(load_checkpoint_path).to(device)
    else:
        model = model_cls(**model_config).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return device, model


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


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
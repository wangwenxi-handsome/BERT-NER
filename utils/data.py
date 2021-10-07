import torch


def dict_to_list_by_max_len(batch):
    keys = list(batch[0].keys())
    batch_values = [list(i.values()) for i in batch]
    batch_values = map(torch.stack, zip(*batch_values))
    batch_dict = dict(zip(keys, batch_values))
    max_len = int(max(batch_dict["length"]).item())
    for i in batch_dict:
        if i != "length":
            batch_dict[i] = batch_dict[i][:, :max_len]
    batch_dict.pop("length")
    return batch_dict
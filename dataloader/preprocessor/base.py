import os
from random import sample
import torch
from torch.utils.data import DataLoader, dataloader
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, List

from dataloader.tokenize import NERTAG, NERTokenize
from utils.torch_related import MyDataSet, dict_to_list_by_max_len


class RDataset:
    """Process and split the data into the form of list.
    """
    def __init__(
        self, 
        ner_tag_method,
        split_rate,
        if_cross_valid = False,
        if_tag_first = True,
    ):
        # ner precess
        self.ner_tag_method = ner_tag_method
        self.if_tag_first = if_tag_first

        # for train split(if no dev)
        self.split_rate = split_rate
        self.cross_vaild = if_cross_valid

    def get_data_with_list_format(self, data):
        """preprocess dataset
        in: [{}, {}, {}...]
            len(in) = the length of raw data.
            {} means each raw data.
        out(get_data): [{"x": [], "y": [], "id": []}, ...]
            len(out) = the splited numbers of data
            {"x": [[], []...], "y": [], "id": []} means one piece of data.
            [[], []...] means the data with list format
        """
        return self._split_data(self._preprocess_data(data))

    def _preprocess_data(self, data):
        """
        in: [{}, {}, {}...]
        out: {"x": , "y": , "id": } (not split)
        """
        raise NotImplementedError(f"please implement the data_precessor func")

    def _split_data(self, data):
        """        
        in: {"x": , "y": , "id": }
        out: [{"x": , "y": , "id": }, ...]
        """
        if len(self.split_rate) == 0:
            splited_data = [data]
        else:
            # first split
            raw_rate = 1
            data_keys = list(data.keys())
            data1 = {}
            data2 = {}
            split_data1 = (train_test_split(
                *(data[i] for i in data),
                test_size = self.split_rate[0], 
                random_state = 42,
            ))
            for i in range(len(split_data1)):
                if i % 2 == 0:
                    data1[data_keys[int(i / 2)]] = split_data1[i]
                else:
                    data2[data_keys[int(i / 2)]] = split_data1[i]
            raw_rate -= self.split_rate[0]

            if len(self.split_rate) == 1:
                splited_data = [data1, data2]
            elif len(self.split_rate) == 2:
                # second split
                data3 = {}
                split_data2 = (train_test_split(
                    *(data1[i] for i in data1),
                    test_size = self.split_rate[1] / raw_rate, 
                    random_state = 42,
                ))
                for i in range(len(split_data2)):
                    if i % 2 == 0:
                        data1[data_keys[int(i / 2)]] = split_data2[i]
                    else:
                        data3[data_keys[int(i / 2)]] = split_data2[i]
                splited_data = [data1, data2, data3]
            else:
                raise ValueError(f"len(split_rate) must <= 2")
        return splited_data

    def get_ner_tag(self):
        return self.ner_tag

    @property
    def classes(self):
        return []


class BasePreProcessor:
    def __init__(
        self,
        rdataset_cls,
        model_name,
        ner_tag_method = "BIO",
        dataloader_name = ["train", "dev", "test"],
        split_rate = [],
        max_length = 256,
    ):
        """before get dataloder, it must call init_data func to initialize the data.
        """
        # weapon prepare!!!
        self.rdataset = rdataset_cls(split_rate = split_rate, ner_tag_method = ner_tag_method)
        self.ner_tag = self.rdataset.get_ner_tag()
        self.tokenize = NERTokenize(ner_tag = self.ner_tag, model_name = model_name, max_length = max_length)

        # dataloader_name decides how many parts data will be divided into.
        self.dataloader_name = dataloader_name
        self.dataloader_name2id = dict(zip(dataloader_name, range(len(self.dataloader_name))))

    def _read_file(self, data_path: str):
        """from data_path to a list of data.
        """
        raise NotImplementedError(f"please implement the _read_file func")
    
    def init_data(self, data_path: Union[str, List[str]]):
        """init the data
        data_path -> [{data1}, {data2}, {data3}...] 
        self.data["list"] -> [{"x": , "y": , "id": }, ...]
        self.data["tensor"] -> [{"input_ids": [], "labels": []...}, {}...]
        """
        if isinstance(data_path, str):
            data_path = [data_path]
        print("www", data_path)
        # 只传入了一个数据集
        if len(data_path) == 1 and data_path[0][-4: ] == ".pth":
            # 是以已经处理过的数据
            print("hhh", data_path)
            data = torch.load(data_path[0])
        # 未处理过
        else:
            data_list = []
            for dp in data_path:
                # read data
                raw_data = self._read_file(dp)
                # to list
                data_list.extend(self.rdataset.get_data_with_list_format(raw_data))
            # 将分好的数据对应到dataloader_name上 
            assert len(data_list) == len(self.dataloader_name)
            data_list = {self.dataloader_name[i]: data_list[i] for i in range(len(data_list))}
            # to tensor
            data_tensor = {}
            for i in data_list:
                data_tensor[i] = self.tokenize.get_data_with_tensor_format(data_list[i])
            data = {"list": data_list, "tensor": data_tensor}
            # save, 取第一个文件的文件名作为名字，但后缀名为.pth
            torch.save(data, os.path.join(os.path.dirname(data_path[0]), "data.pth"))
        self.data = data

    def get_dataloader(
        self, 
        batch_size, 
        if_DDP_mode,
        num_workers=0, 
        collate_fn=dict_to_list_by_max_len,
    ):
        dataloader = {}
        sampler = None
        for i in self.data["tensor"]:
            dataset = MyDataSet(**self.data["tensor"][i])
            if i == "train":
                shuffle = True
                if if_DDP_mode:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                shuffle = False

            dataloader[i] = DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle = shuffle, 
                num_workers = num_workers,
                collate_fn = collate_fn,
                sampler = sampler,
            )
        return dataloader

    def get_raw_data_x(self, name):
        return self.data["list"][name]["x"]

    def get_raw_data_y(self, name):
        return self.data["list"][name]["y"]

    def get_raw_data_id(self, name):
        return self.data["list"][name]["id"]

    def get_tokenize_length(self, name):
        return self.data["tensor"][name]["length"]

    def get_ner_tag(self):
        return self.ner_tag

    def decode(self, outputs, tokenize_length, labels = None, offset_mapping = None):
        return self.tokenize.decode(outputs, tokenize_length, labels, offset_mapping)
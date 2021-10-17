import os
import torch
from torch.utils.data import DataLoader, dataloader
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, List

from dataloader.tokenize import NERTAG, NERTokenize
from utils.torch_related import MyDataSet, dict_to_list_by_max_len


class RDataset:
    def __init__(
        self, 
        split_rate = [],
        if_cross_valid = False,
        ner_tag_method = "BIO",
        cased = True,
        if_tag_first = True,
    ):
        # ner precess
        self.cased = cased
        self.ner_tag_method = ner_tag_method
        self.if_tag_first = if_tag_first

        # for train split(if no dev)
        self.split_rate = split_rate
        self.cross_vaild = if_cross_valid
    
    def get_ner_tag(self):
        return self.ner_tag

    def get_data_with_list_format(self, data):
        """preprocess dataset
        in: [{}, {}, {}...]
        out(get_data): [{"x": , "y": , "id": }, ...]
        """
        return self.split_data(self.preprocess_data(data))

    def split_data(self, data):
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
            ()
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

    def preprocess_data(self, data):
        """
        in: [{}, {}, {}...]
        out: {"x": , "y": , "id": } (not split)
        """
        raise NotImplementedError(f"please implement the data_precessor func")

    @property
    def classes(self):
        return []


class BasePreProcessor:
    def __init__(
        self,
        rdataset_cls,
        model_name,
        data_path: Union[str, List[str]],
        dataloader_name = ["train", "dev", "test"],
        split_rate = [],
    ):
        # weapon prepare!!!
        self.rdataset = rdataset_cls(split_rate = split_rate)
        self.ner_tag = self.rdataset.get_ner_tag()
        self.tokenize = NERTokenize(ner_tag = self.ner_tag, model_name = model_name)

        # raw_data -> [{data1}, {data2}, {data3}...] 
        # self.data["list"] -> [{"x": , "y": , "id": }, ...]
        # self.data["tensor"] -> [{"input_ids": [], "labels": []...}, {}...]
        # self.data = self.init_data(data_path)

        # dataloader_name decides how many parts data will be divided into.
        self.dataloader_name = dataloader_name
        self.dataloader_name2id = dict(zip(dataloader_name, range(len(self.dataloader_name))))
        self.data = None
        self.data_path = data_path
    
    def init_data(self, data_path):
        if isinstance(data_path, str):
            data_path = [data_path]
        # 只传入了一个数据集
        if len(data_path) == 1 and data_path[0][-4: ] == ".pth":
            # 是以已经处理过的数据
            data = torch.load(data_path[0])
        # 未处理过
        else:
            data_list = []
            for dp in data_path:
                # read data
                raw_data = self.read_file(dp)
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
            torch.save(data, os.path.join(os.path.dirname(data_path), "data.pth"))
        return data

    def get_dataloader(self, batch_size, num_workers=0, collate_fn=dict_to_list_by_max_len):
        dataloader = {}
        for i in self.data["tensor"]:
            if i == "train":
                shuffle = True
            else:
                shuffle = False
            dataloader[i] = DataLoader(
                MyDataSet(**self.data["tensor"][i]), 
                batch_size = batch_size, 
                shuffle = shuffle, 
                num_workers = num_workers,
                collate_fn = collate_fn,
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

    def read_file(self, data_path):
        raise NotImplementedError(f"please implement the read_file func")

    def decode(self, outputs, tokenize_length, labels = None, offset_mapping = None):
        return self.tokenize.decode(outputs, tokenize_length, labels, offset_mapping)
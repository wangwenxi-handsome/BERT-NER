import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.torch_related import MyDataSet, dict_to_list_by_max_len
from dataloader.tokenize import NERTAG, NERTokenize
from dataloader.preprocessor.base import RDataset


class BYTERDataset(RDataset):
    def __init__(
        self, 
        split_rate = [],
        if_cross_valid = False,
        ner_tag_method = "BIO",
        cased = True,
        if_tag_first = True,
    ):
        super(BYTERDataset, self).__init__(split_rate, if_cross_valid, ner_tag_method, cased, if_tag_first)
        self.ner_tag = NERTAG(self.classes, ner_tag_method, if_tag_first)

    def preprocess_data(self, data):
        new_data = {"x": [], "y": [], "id": []}
        for d in data:
            d = eval(d)
            now_sentence = list(d["sentence"])
            now_label = ["O" for _ in range(len(now_sentence))]
            for i in d["results"]:
                start = i[0]
                end = i[1]
                ner_class = i[2]
                for j in range(start, end):
                    if j == start:
                        now_label[j] = "B-" + ner_class
                    else:
                        now_label[j] = "I-" + ner_class
            new_data["x"].append(now_sentence)
            now_label = [self.ner_tag.tag2id[w] for w in now_label]
            new_data["y"].append(now_label)
            new_data["id"].append(d["itemID"])
        return new_data

    @property
    def classes(self):
        return [
            'other', '事件-other', '事件-节假日', '事件-行业会议', '事件-项目策划', '产品-other',
            '产品-交通工具', '产品-文娱产品', '产品-服饰', '产品-设备工具', '产品-金融产品', '产品-食品', '地点-other', 
            '地点-公共场所', '地点-楼宇建筑物', '技术术语-技术指标', '技术术语-技术标准', '技术术语-技术概念', '组织-other', 
            '组织-企业机构', '组织-科研院校', '组织-行政机构', '组织-部门团体', '职业岗位', '规定-other', '规定-法律法规', 
            '规定-规章制度', '软件系统-other', '软件系统-应用软件', '软件系统-系统平台', '软件系统-网站'
        ]


class BYTEPreProcessor:
    def __init__(
        self,
        data_path,
        model_name,
        split_rate = [0.1, 0.1],
    ):
        # weapon prepare!!!
        self.rdataset = BYTERDataset(split_rate = split_rate)
        self.ner_tag = self.rdataset.get_ner_tag()
        self.tokenize = NERTokenize(ner_tag = self.ner_tag, model_name = model_name, return_offsets_mapping = False)

        # self.data["raw"] -> [{data1}, {data2}, {data3}...] 
        # self.data["list"] -> [{"x": , "y": , "id": }, ...]
        # self.data["tensor"] -> [{"input_ids": [], "labels": []...}, {}...]
        self.data = self.init_data(data_path)
        self.dataloader_name = ["train", "dev", "test"]
        self.dataloader_name2id = dict(zip(self.dataloader_name, range(len(self.dataloader_name))))
    
    def init_data(self, data_path):
        if data_path[-4: ] == ".pth":
            data = torch.load(data_path)
        else:
            assert data_path[-4: ] == ".npy"
            # read data
            raw_data = self.read_file(data_path)
            # to list
            data_list = self.rdataset.get_data_with_list_format(raw_data)
            # to tensor
            data_tensor = []
            for i in data_list:
                data_tensor.append(self.tokenize.get_data_with_tensor_format(i))
            data = {"raw": raw_data, "list": data_list, "tensor": data_tensor}
            torch.save(data, data_path[: -4] + ".pth")
        return data

    def read_file(self, data_path):
        return np.load(data_path).tolist()

    def get_dataloader(self, batch_size, num_workers):
        dataloader = {}
        for i in range(len(self.dataloader_name)):
            if self.dataloader_name[i] == "train":
                shuffle = True
            else:
                shuffle = False
            dataloader[self.dataloader_name[i]] = DataLoader(
                MyDataSet(**self.data["tensor"][i]), 
                batch_size = batch_size, 
                shuffle = shuffle, 
                num_workers = num_workers,
                collate_fn = dict_to_list_by_max_len,
            )
        return dataloader

    def get_raw_data_x(self, name):
        return self.data["list"][self.dataloader_name2id[name]]["x"]

    def get_raw_data_y(self, name):
        return self.data["list"][self.dataloader_name2id[name]]["y"]

    def get_raw_data_id(self, name):
        return self.data["list"][self.dataloader_name2id[name]]["id"]

    def get_tokenize_length(self, name):
        return self.data["tensor"][self.dataloader_name2id[name]]["length"]

    def get_ner_tag(self):
        return self.ner_tag
import numpy as np

from dataloader.tokenize import NERTAG
from dataloader.preprocessor.base import RDataset, BasePreProcessor


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
            # !!!TODO why is dict
            if isinstance(d, str):
                d = eval(d)
            now_sentence = list(d["sentence"])
            now_label = ["O" for _ in range(len(now_sentence))]

            # 通过异常处理添加y
            try:
                for i in d["results"]:
                    start = i[0]
                    end = i[1]
                    ner_class = i[2]
                    for j in range(start, end):
                        if j == start:
                            now_label[j] = "B-" + ner_class
                        else:
                            now_label[j] = "I-" + ner_class
                now_label = [self.ner_tag.tag2id[w] for w in now_label]
                new_data["y"].append(now_label)
            except:
                if "y" in new_data:
                    new_data.pop("y")
            
            # 添加x和id
            new_data["x"].append(now_sentence)
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


class BYTEPreProcessor(BasePreProcessor):
    def __init__(
        self,
        model_name,
        data_path,
        split_rate = [0.1, 0.1],
    ):
        super(BYTEPreProcessor, self).__init__(
            rdataset_cls=BYTERDataset,
            model_name = model_name,
            data_path = data_path,
            dataloader_name = ["train", "dev", "test"],
            split_rate = split_rate,
        )
        self.data = self.init_data(data_path)

    def read_file(self, data_path):
        return np.load(data_path).tolist()


class BYTETESTPreProcessor(BasePreProcessor):
    def __init__(
        self,
        model_name,
        data_path,   
    ):
        super(BYTETESTPreProcessor, self).__init__(
            rdataset_cls=BYTERDataset,
            model_name = model_name,
            data_path = data_path,
            dataloader_name = ["test"],
            split_rate = [],
        )
        self.data = self.init_data(data_path)

    # TEST时候的data_path直接就是data的形式
    def init_data(self, data_path):
        data_list = []
        data_list.extend(self.rdataset.get_data_with_list_format(data_path))
        # 将分好的数据对应到dataloader_name上 
        assert len(data_list) == len(self.dataloader_name)
        data_list = {self.dataloader_name[i]: data_list[i] for i in range(len(data_list))}
        # to tensor
        data_tensor = {}
        for i in data_list:
            data_tensor[i] = self.tokenize.get_data_with_tensor_format(data_list[i])
        data = {"list": data_list, "tensor": data_tensor}
        return data


class BYTETrainPreProcessor(BasePreProcessor):
    def __init__(
        self,
        model_name,
        data_path,   
    ):
        super(BYTETrainPreProcessor, self).__init__(
            rdataset_cls=BYTERDataset,
            model_name = model_name,
            data_path = data_path,
            dataloader_name = ["train", "dev"],
            split_rate = [0.1],
        )
        self.data = self.init_data(data_path)

    # Train时候的data_path直接就是data的形式
    def init_data(self, data_path):
        data_list = []
        data_list.extend(self.rdataset.get_data_with_list_format(data_path))
        # 将分好的数据对应到dataloader_name上 
        assert len(data_list) == len(self.dataloader_name)
        data_list = {self.dataloader_name[i]: data_list[i] for i in range(len(data_list))}
        # to tensor
        data_tensor = {}
        for i in data_list:
            data_tensor[i] = self.tokenize.get_data_with_tensor_format(data_list[i])
        data = {"list": data_list, "tensor": data_tensor}
        return data
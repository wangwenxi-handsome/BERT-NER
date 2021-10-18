import os

from dataloader.tokenize import NERTAG
from dataloader.preprocessor.base import RDataset, BasePreProcessor


class CNERRDataset(RDataset):
    def __init__(
        self, 
        ner_tag_method,
        split_rate = [],
    ):
        super(CNERRDataset, self).__init__(split_rate = split_rate, ner_tag_method = ner_tag_method)
        self.ner_tag = NERTAG(self.classes, ner_tag_method)

    def _preprocess_data(self, data):
        new_data = {"x": [], "y": [], "id": []}
        new_data["x"] = data[0]
        new_data["y"] = self._add_ner_tag(data[1])
        new_data["id"] = list(range(len(new_data["x"])))
        return new_data

    def _add_ner_tag(self, data_y):
        new_data_y = []
        for i in data_y:
            tmp_data_y = []
            if self.ner_tag_method == "BIO":
                for j in i:
                    if j[0] == "M":
                        tmp_data_y.append(self.ner_tag.tag2id["I" + j[1:]])
                    elif j[0] == "S":
                        tmp_data_y.append(self.ner_tag.tag2id["B" + j[1:]])
                    elif j[0] == "E":
                        tmp_data_y.append(self.ner_tag.tag2id["I" + j[1:]])
                    else:
                        tmp_data_y.append(self.ner_tag.tag2id[j])
            elif self.ner_tag_method == "BIOS":
                for j in i:
                    if j[0] == "M":
                        tmp_data_y.append(self.ner_tag.tag2id["I" + j[1:]])
                    elif j[0] == "E":
                        tmp_data_y.append(self.ner_tag.tag2id["I" + j[1:]])
                    else:
                        tmp_data_y.append(self.ner_tag.tag2id[j])
            elif self.ner_tag_method == "BMESO":
                tmp_data_y = [self.ner_tag.tag2id[j] for j in i]
            else:
                raise NotImplementedError(f"please implemnt add_ner_tag func for {self.ner_tag_method}")
            new_data_y.append(tmp_data_y)
        return new_data_y
    
    @property
    def classes(self):
        return ['EDU', 'TITLE', 'PRO', 'ORG', 'LOC', 'NAME', 'CONT', 'RACE']


class CNERPreProcessor(BasePreProcessor):
    def __init__(
        self,
        model_name,
        ner_tag_method = "BIOS",
        train_fn = "train.char.bmes",
        dev_fn = "dev.char.bmes",
        test_fn = "test.char.bmes",
    ):
        super(CNERPreProcessor, self).__init__(
            rdataset_cls = CNERRDataset,
            model_name = model_name,
            ner_tag_method = ner_tag_method,
            dataloader_name = ["train", "dev", "test"],
            split_rate = [],
        )
        self.train_fn = train_fn
        self.dev_fn = dev_fn
        self.test_fn = test_fn

    def init_data(self, folder_name: str):
        if folder_name[-4:] == ".pth":
            data_path = folder_name
        else:
            data_path = [
                os.path.join(folder_name, self.train_fn), 
                os.path.join(folder_name, self.dev_fn), 
                os.path.join(folder_name, self.test_fn), 
            ]
        super(CNERPreProcessor, self).init_data(data_path)

    def _read_file(self, data_path):
        data_x = []
        data_y = []
        tmp_x = []
        tmp_y = []
        with open(data_path,'r') as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tmp_x:
                        assert len(tmp_x) == len(tmp_y)
                        data_x.append(tmp_x)
                        data_y.append(tmp_y)
                        tmp_x = []
                        tmp_y = []
                else:
                    splits = line.split(" ")
                    tmp_x.append(splits[0])
                    assert len(splits) > 1
                    tmp_y.append(splits[-1].replace("\n", ""))
            if tmp_x:
                assert len(tmp_x) == len(tmp_y)
                data_x.append(tmp_x)
                data_y.append(tmp_y)
        return [data_x, data_y]
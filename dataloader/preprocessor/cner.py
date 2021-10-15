from dataloader.processor.ner_processor import NERTAG
from dataloader.dataset.base import BaseDataset


class CNER(BaseDataset):
    def __init__(
        self, 
        ner_tag_method = "BMESO",
        base_config = {},
    ):
        super(CNER, self).__init__(**base_config)
        self.is_split_into_words = True
        self.ner_tag_method = ner_tag_method

    def data_precessor(self, folder_path):
        # json to list
        data_x, data_y = self.bmes_to_list(folder_path)
        self.ner_tag = NERTAG(self.get_labels(data_y), self.ner_tag_method, self.if_tag_first)
        data_y = self.add_ner_tag(data_y)
        return data_x, data_y

    def bmes_to_list(self, folder_path):
        data_x = []
        data_y = []
        tmp_x = []
        tmp_y = []
        with open(folder_path,'r') as f:
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
        return data_x, data_y
        
    def get_labels(self, data_y):
        labels = set()
        for i in data_y:
            now_labels = []
            for j in i:
                if "-" in j:
                    now_labels.append(j.split("-")[1])
            labels = labels | set(now_labels)
        return list(labels)

    def add_ner_tag(self, data_y):
        new_data_y = []
        for i in data_y:
            tmp_data_y = [self.ner_tag.tag2id[j] for j in i]
            new_data_y.append(tmp_data_y)
        return new_data_y
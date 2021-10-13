import os
import numpy as np

from dataloader.processor.ner_processor import NERTAG
from dataloader.dataset.base import BaseDataset


class BYTENER(BaseDataset):
    def __init__(
        self, 
        ner_tag_method = "BIO",
        base_config = {},
    ):
        super(BYTENER, self).__init__(**base_config)
        self.is_split_into_words = True
        self.ner_tag_method = ner_tag_method

    def data_precessor(self, folder_path):
        data = np.load(folder_path).tolist()
        data_x, data_y, self.all_labels = self.pick_ner_item(data)
        self.ner_tag = NERTAG(self.all_labels, self.ner_tag_method, self.if_tag_first)
        for i in range(len(data_y)):
            data_y[i] = [self.ner_tag.tag2id[j] for j in data_y[i]]
        return data_x, data_y

    def pick_ner_item(self, data):
        all_labels = set()
        data_x = []
        data_y = []
        for d in data:
            d = eval(d)
            now_sentence = list(d["sentence"])
            now_label = ["O" for _ in range(len(now_sentence))]
            for i in d["results"]:
                start = i[0]
                end = i[1]
                ner_class = i[2]
                all_labels.add(ner_class)
                for j in range(start, end):
                    if j == start:
                        now_label[j] = "B-" + ner_class
                    else:
                        now_label[j] = "I-" + ner_class
            data_x.append(now_sentence)
            data_y.append(now_label)
        return data_x, data_y, all_labels

    def ner_list_sub_int(self, ner_list, word_sum):
        _ner_list = []
        for ner in ner_list:
            ner0 = ner[0] - word_sum
            ner1 = ner[1] - word_sum
            _ner_list.append([ner0, ner1, ner[2]])
        return _ner_list
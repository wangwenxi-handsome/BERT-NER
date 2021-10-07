import os
import json
import numpy as np

from dataloader.ner_tag import NERTAG


class ACE05:
    def __init__(
        self, 
        task = "NER",
        data_path = "/Users/bytedance/Desktop/nlp-dataset/ace05/json",
        train_fn = "train.json",
        test_fn = "test.json",
        dev_fn = "dev.json",
        ner_tag_method = "BIO",
        cased = True,
    ):
        # property
        self.is_split_into_words = True

        # data path
        self.data_path = data_path
        self.train_fn = os.path.join(data_path, train_fn)
        self.test_fn = os.path.join(data_path, test_fn)
        self.dev_fn = os.path.join(data_path, dev_fn)

        # common precess
        self.cased = cased

        # task
        self.task = task
        if self.task == "NER":
            # NER
            self.ner_tag = NERTAG(['PER', 'FAC', 'ORG', 'WEA', 'GPE', 'LOC', 'VEH'], ner_tag_method)
        else:
            raise NotImplementedError(f"{task} is not implemented")

    def get_data(self):
        save_file_path = os.path.join(self.data_path, "data.npy")
        if os.path.exists(save_file_path):
            data = np.load(save_file_path, allow_pickle=True).item()
        else:
            train_x, train_y = self.data_precessor(self.train_fn)
            dev_x, dev_y = self.data_precessor(self.dev_fn)
            test_x, test_y = self.data_precessor(self.test_fn)
            data = {"train": {"x": train_x, "y": train_y}, "dev": {"x": dev_x, "y": dev_y}, "test": {"x": test_x, "y": test_y}}
            np.save(save_file_path, data)
        return data

    def data_precessor(self, data_path):
        # json to list
        data = self.json_to_list(data_path)
        # get data of the specific task
        if self.task == "NER":
            data_x, data_y = self.pick_ner_item(data)
            # tag for NER
            data_x, data_y = self.add_ner_tag(data_x, data_y)
        else:
            raise NotImplementedError(f"please implement data_precessor for {self.task}")
        return data_x, data_y

    def json_to_list(self, data_path):
        data = []
        with open(data_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data.append(json.loads(line))
        return data

    def pick_ner_item(self, data):
        data_x = []
        data_y = []
        for d in data:
            word_sum = 0
            assert len(d["ner"]) == len(d["sentences"])
            for id in range(len(d["ner"])):
                if len(d["ner"][id]) > 0:
                    if not self.cased:
                        d["sentences"][id] = [s.lower() for s in d["sentences"][id]]
                    data_x.append(d["sentences"][id])
                    data_y.append(self.ner_list_sub_int(d["ner"][id], word_sum))
                word_sum += len(d["sentences"][id])
        return data_x, data_y

    def ner_list_sub_int(self, ner_list, word_sum):
        _ner_list = []
        for ner in ner_list:
            ner0 = ner[0] - word_sum
            ner1 = ner[1] - word_sum
            _ner_list.append([ner0, ner1, ner[2]])
        return _ner_list

    def add_ner_tag(self, data_x, data_y):
        new_data_y = []
        for i in range(len(data_x)):
            sentence = data_x[i]
            if self.ner_tag.ner_tag_method == "BIO":
                tag = [self.ner_tag.tag2id["O"] for _ in range(len(sentence))]
                for ner in data_y[i]:
                    tag[ner[0]] = self.ner_tag.tag2id[ner[2] + "-B"]
                    if ner[1] > ner[0]:
                        tag[ner[1]] = self.ner_tag.tag2id[ner[2] + "-I"]
                new_data_y.append(tag)
            else:
                raise NotImplementedError(f"please implement {self.ner_tag.ner_tag_method} for ner_tag")
        return data_x, new_data_y
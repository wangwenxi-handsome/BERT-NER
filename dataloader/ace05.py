import os
import torch
import torch.utils.data as Data
import json
from sentence_transformers import SentenceTransformer


class ACE05:
    def __init__(
        self, 
        data_path = "/Users/bytedance/Desktop/nlp-dataset/ace05/json",
        train_fn = "train.json",
        test_fn = "test.json",
        dev_fn = "dev.json",
        sequence_length = 128,
        tag_method = "BIO",
    ):
        self.tag = ['PER', 'FAC', 'ORG', 'WEA', 'GPE', 'LOC', 'VEH']
        self.tag2id = dict(zip(self.tag, [i for i in range(len(self.tag))]))
        self.method2id = {"B": 0, "I": 1, "O": 2}
        self.tag_method = "BIO"
        self.train_fn = os.path.join(data_path, train_fn)
        self.test_fn = os.path.join(data_path, test_fn)
        self.dev_fn = os.path.join(data_path, dev_fn)
        self.sq_len = sequence_length
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        self.precessor()

    def map_tag_to_number(self, tag):
        number = [0 * 15]
        if tag == "O":
            number[0] = 1
            return number
        a, b = tag.split("-")
        number[self.tag2id[a] * 2 + self.method2id[b] + 1] = 1
        return number

    def precessor(self, ):
        # get all item
        self.train_x, self.train_y = self.json_to_list(self.train_fn)
        self.test_x, self.test_y = self.json_to_list(self.test_fn)
        self.dev_x, self.dev_y = self.json_to_list(self.dev_fn)

        self.train_x = [self.sentence_transformer(x) for x in self.train_x]
        self.test_x = [self.sentence_transformer(x) for x in self.test_x]
        self.dev_x = [self.sentence_transformer(x) for x in self.dev_x]

    def json_to_list(self, data_path):
        data = []
        with open(data_path) as f:
            while True:
                line = f.readline()
                if not line: # 到 EOF，返回空字符串，则终止循环
                    break
                data.append(json.loads(line))

        data_x = []
        data_y = []
        for d in data:
            word_sum = 0
            assert len(d["ner"]) == len(d["sentences"])
            for id in range(len(d["ner"])):
                if len(d["ner"][id]) > 0:
                    data_x.append(d["sentences"][id])
                    data_y.append(self.list_sub_int(d["ner"][id], word_sum))
                word_sum += len(d["sentences"][id])
        return self.eio_tag(data_x, data_y)

    def ner_list_sub_int(self, ner_list, word_sum):
        _ner_list = []
        for ner in ner_list:
            ner0 = ner[0] - word_sum
            ner1 = ner[1] - word_sum
            _ner_list.append([ner0, ner1, ner[2]])
        return _ner_list

    def eio_tag(self, data_x, data_y):
        new_data_y = []
        for i in range(len(data_x)):
            sentence = data_x[i]
            assert len(sentence) <= self.sq_len
            tag = [self.map_tag_to_number["O"] for word in self.sq_len]
            for ner in data_y[i]:
                tag[ner[0]] = self.map_tag_to_number(ner[2] + "-B")
                if ner[1] > ner[0]:
                    tag[ner[1]] = self.map_tag_to_number(ner[2] + "-I")
            new_data_y.append(tag)
        return data_x, new_data_y


def wrap_dataloader():
    data = ACE05()
    
    train_x, train_y = torch.tensor(data.train_x), torch.tensor((data.train_y))
    test_x, test_y = torch.tensor(data.test_x), torch.tensor((data.test_y))
    dev_x, dev_y = torch.tensor(data.dev_x), torch.tensor((data.dev_y))

    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    dev_dataset = Data.TensorDataset(dev_x, dev_y)

    train_dataloader = Data.DataLoader(train_dataset, 24, False, 4)
    test_dataloader = Data.DataLoader(test_dataset, 24, False, 4)
    dev_dataloader = Data.DataLoader(dev_dataset, 24, False, 4)

    return train_dataloader, test_dataloader, dev_dataloader


    

    
        
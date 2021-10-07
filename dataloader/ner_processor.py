import os
import torch
from torch._C import dtype
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class NERDataSet(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return self.kwargs[list(self.kwargs.keys())[0]].size()[0]

    def __getitem__(self, id):
        return {i: self.kwargs[i][id] for i in self.kwargs}


class NERProcessor:
    def __init__(
        self,
        data_cls,
        data_config,
        tokenizer_model = "bert-base-cased",
        tokenizer_cls = BertTokenizerFast,
        is_split_into_words = False,
        return_offsets_mapping = True,
        padding = "max_length",
        truncation = True,
        max_length = None,
        return_tensors = "pt",
        language = "en",
    ):
        self.data_cls = data_cls(**data_config)
        self.ner_tag = self.data_cls.ner_tag
        self.language = language

        # tokenizer related
        self.tokenizer = tokenizer_cls.from_pretrained(tokenizer_model)
        self.is_split_into_words = is_split_into_words
        self.return_offsets_mapping = return_offsets_mapping
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.return_tensors = return_tensors

    def get_tensor(self):
        save_file_path = os.path.join("/opt/tiger/nlp-xixi/data", "data.pth")
        if os.path.exists(save_file_path):
            return torch.load(save_file_path)
        else:
            self.data = self.data_cls.get_data()
            for item in self.data:
                data_x = self.tokenizer(
                    self.data[item]["x"],
                    is_split_into_words = self.is_split_into_words, 
                    return_offsets_mapping= self.return_offsets_mapping,
                    padding = self.padding,
                    truncation = self.truncation,
                    max_length = self.max_length,
                    return_tensors = self.return_tensors,
                )
                data_y, data_len = self.get_modified_tag_and_len(data_x, self.data[item]["y"])
                data = {
                    "input_ids": data_x["input_ids"],
                    "token_type_ids": data_x["token_type_ids"],
                    "attention_mask": data_x["attention_mask"],
                    "offset_mapping": data_x["offset_mapping"],
                    "label": torch.tensor(data_y, dtype=torch.long),
                    "length": torch.tensor(data_len, dtype=torch.long),
                }
                self.data[item] = data
            torch.save(self.data, save_file_path)
            return self.data

    def get_modified_tag_and_len(self, data_x, data_y):
        new_data_y = []
        data_len = []
        for i in range(len(data_y)):
            # 起止标志
            now_data_y = [0] + data_y[i] + [0]

            # 由于一个词被划分成多个导致的label偏移
            if self.return_offsets_mapping:
                now_word = 0
                now_id = 0
                offset = data_x["offset_mapping"][i]
                for j in range(len(offset)):
                    if offset[j][0] == 0:
                        if now_word >= len(now_data_y):
                            data_len.append(len(now_data_y))
                            break
                        now_id = now_data_y[now_word]
                        now_word += 1
                    else:
                        now_data_y = now_data_y[: now_word] + [self.ner_tag.map_B2I(now_id)] + now_data_y[now_word: ]
                        now_word += 1

            # padding
            now_data_y += [0] * (len(data_x["input_ids"][i]) - len(now_data_y))
            new_data_y.append(now_data_y)
        return new_data_y, data_len
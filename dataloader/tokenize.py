import torch
from transformers import BertTokenizerFast


class NERTAG:
    def __init__(
        self,
        ner_class,
        ner_tag_method,
        if_tag_first = True,
    ):
        self.ner_class = ner_class
        self.ner_tag_method = ner_tag_method
        self.if_tag_first = if_tag_first

        self.tag2id = {"O": 0}
        i = 0
        for c in self.ner_class:
            for s in self.ner_tag_method:
                if s == "O":
                    continue
                else:
                    i += 1
                    if self.if_tag_first:
                        self.tag2id[s + "-" + c] = i
                    else:
                        self.tag2id[c + "-" + s] = i
        self.id2tag = dict(zip(self.tag2id.values(), self.tag2id.keys()))

    def map_B2I(self, id):
        if id == 0:
            return 0
        tag = self.id2tag[id]
        if self.if_tag_first:
            if tag[0] == "B":
                tag = "I" + tag[1: ]
        else:
            if tag[-1] == "B":
                tag = tag[:-1] + "I"
        return self.tag2id[tag]

    def __len__(self):
        return len(self.id2tag)


class NERTokenize:
    def __init__(
        self,
        ner_tag,
        model_name = "bert-base-cased",
        tokenizer_cls = BertTokenizerFast,
        return_offsets_mapping = True,
        padding = "max_length",
        truncation = True,
        max_length = None,
    ):
        self.ner_tag = ner_tag

        # tokenizer related
        self.tokenizer = tokenizer_cls.from_pretrained(model_name)
        self.return_offsets_mapping = return_offsets_mapping
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def get_data_with_tensor_format(self, data):
        """get data with tensor format
        in: {"x": , "y": , "id": }
        out: {"input_ids":, "token_type_ids":, "attention_mask":, "offset_mapping":, "labels":, "length":}
        """
        data_x = self.tokenizer(
            data["x"],
            is_split_into_words = True, 
            return_offsets_mapping= True,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length,
            return_tensors = "pt",
        )
        data_y, data_len = self.get_modified_tag_and_len(data_x, data["y"])
        new_data = {
            "input_ids": data_x["input_ids"],
            "token_type_ids": data_x["token_type_ids"],
            "attention_mask": data_x["attention_mask"],
            "labels": torch.tensor(data_y, dtype=torch.long),
            "length": torch.tensor(data_len, dtype=torch.long),
        }
        # return offset mapping
        if self.return_offsets_mapping:
            new_data["offset_mapping"] = new_data["offset_mapping"],
        else:
            new_data["offset_mapping"] = None
        return new_data

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
                        now_data_y = now_data_y[: now_word] + [self.data_cls.ner_tag.map_B2I(now_id)] + now_data_y[now_word: ]
                        now_word += 1
            else:
                data_len.append(len(now_data_y))

            # padding
            now_data_y += [0] * (len(data_x["input_ids"][i]) - len(now_data_y))
            new_data_y.append(now_data_y)
        return new_data_y, data_len
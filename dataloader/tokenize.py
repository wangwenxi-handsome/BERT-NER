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

    def __len__(self):
        return len(self.id2tag)

    def map_tag2id(self, tag):
        return self.tag2id[tag]

    def map_id2tag(self, id):
        return self.id2tag[id]

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

    def get_ner_tag_method(self):
        return self.ner_tag_method

    def get_if_tag_first(self):
        return self.if_tag_first


class NERTokenize:
    def __init__(
        self,
        ner_tag,
        model_name,
        cased = True,
        tokenizer_cls = BertTokenizerFast,
        padding = "max_length",
        truncation = True,
        max_length = 256,
    ):
        self.ner_tag = ner_tag

        # tokenizer related
        self.tokenizer = tokenizer_cls.from_pretrained(model_name, do_lower_case = cased)
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
        data_len = self._get_tokenize_length(data["x"], data_x["offset_mapping"])
        new_data = {
            "input_ids": data_x["input_ids"],
            "token_type_ids": data_x["token_type_ids"],
            "attention_mask": data_x["attention_mask"],
            "offset_mapping": data_x["offset_mapping"],
            "length": torch.tensor(data_len, dtype=torch.long),
        }
        # if there is y
        if "y" in data:
            data_y = self._get_modified_tag(data_x, data["y"])
            new_data["labels"] = torch.tensor(data_y, dtype=torch.long)
        return new_data

    def _get_modified_tag(self, data_x, data_y):
        new_data_y = []
        for i in range(len(data_y)):
            # 起止标志
            now_data_y = [0] + data_y[i] + [0]

            # 由于一个词被划分成多个导致的label偏移
            now_word = 0
            now_id = 0
            offset = data_x["offset_mapping"][i]
            for j in range(len(offset)):
                if offset[j][0] == 0:
                    if now_word >= len(now_data_y):
                        break
                    now_id = now_data_y[now_word]
                    now_word += 1
                else:
                    now_data_y = now_data_y[: now_word] + [self.ner_tag.map_B2I(now_id)] + now_data_y[now_word: ]
                    now_word += 1

            # padding
            now_data_y += [0] * (len(data_x["input_ids"][i]) - len(now_data_y))
            # trunc
            now_data_y = now_data_y[:len(data_x["input_ids"][i])]
            new_data_y.append(now_data_y)
        return new_data_y

    def _get_tokenize_length(self, data, offset):
        data_len = []
        for i in range(len(data)):
            # 首尾标记
            length = len(data[i]) + 2
            # 多分出来一个新词则长度+1
            for o in offset[i]:
                if o[0] != 0:
                    length += 1
            data_len.append(length)
        return data_len

    # 将输出的tensor和labels转换成[[(class, start, end), ()...], [句子2], [句子3]...]的形式
    def decode(self, outputs, tokenize_length, labels = None, offset_mapping = None):
        # process outputs
        # argmax
        outputs = [torch.argmax(i, dim = -1).numpy().tolist() for i in outputs]
        # flatten batch
        new_outputs = []
        for i in outputs:
            new_outputs.extend(i)
        # length
        for i in range(len(new_outputs)):
            new_outputs[i] = new_outputs[i][: tokenize_length[i]]
        # offset
        offset_outputs = self._offset(new_outputs, offset_mapping)
        # tag to entity
        entity_outputs = self._change_tag2entity(offset_outputs)

        # process labels
        if labels is not None:
            entity_labels = self._change_tag2entity(labels)
        else:
            entity_labels = None
        return entity_outputs, entity_labels 

    def _offset(self, outputs, offset_mapping):
        new_outputs = []
        if offset_mapping is None:
            for i in outputs:
                # 掐头去尾
                new_outputs.append(i[1: -1])
        else:
            offset_mapping = offset_mapping.numpy().tolist()
            for i in range(len(outputs)):
                item = outputs[i]
                offs = offset_mapping[i]
                now_outputs = []
                j = 1
                # 取出表示同一个单词的tag
                while(j < len(item) - 1):
                    all_label_for_one_word = [item[j]]
                    j += 1
                    while(offs[j][0] != 0):
                        all_label_for_one_word.append(item[j])
                        j += 1
                    now_outputs.append(self._agg_all_label_for_one_word(all_label_for_one_word))
                new_outputs.append(now_outputs)
        return new_outputs

    def _agg_all_label_for_one_word(self, label_list):
        for i in label_list:
            if i != 0:
                return i
        return 0

    def _change_tag2entity(self, data_y):
        all_entity = []
        for sentence in data_y:
            sentence_entity = []
            sentence = [self.ner_tag.map_id2tag(i) for i in sentence]
            w = 0
            while(w < len(sentence)):
                if self.ner_tag.get_ner_tag_method() == "BIO" or self.ner_tag.get_ner_tag_method() == "BIOS":
                    if self.ner_tag.get_if_tag_first():
                        if sentence[w][0] == "B":
                            now_class = sentence[w][2:]
                            start = w
                            w = w + 1
                            while(w < len(sentence) and sentence[w] == "I-" + now_class):
                                w += 1
                            sentence_entity.append((now_class, start, w - 1))
                        elif sentence[w][0] == "S":
                            now_class = sentence[w][2:]
                            sentence_entity.append((now_class, w, w))
                            w += 1
                        else:
                            w += 1
                    else:
                        if sentence[w][-1] == "B":
                            now_class = sentence[w][:-2]
                            start = w
                            w = w + 1
                            while(w < len(sentence) and sentence[w] == now_class + "-I"):
                                w += 1
                            sentence_entity.append((now_class, start, w - 1))
                        elif sentence[w][-1] == "S":
                            now_class = sentence[w][:-2]
                            sentence_entity.append((now_class, w, w))
                            w += 1
                        else:
                            w += 1
                else:
                    raise NotImplementedError(f"please implement the {self.ner_tag.get_ner_tag_method()} method")
            all_entity.append(sentence_entity)
        return all_entity
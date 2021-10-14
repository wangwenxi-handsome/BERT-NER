import torch
from collections import Counter
from sklearn.metrics import classification_report


class NERMetric:
    def __init__(self, sequence, label, output, length, ner_tag, offset_mapping = None):
        self.ner_tag = ner_tag
        self.sequence = sequence
        self.label = label

        # process output
        self.tmp_output = self.argmax(output, length.numpy().tolist())
        # 处理由于tokenize导致的偏移和占位符
        self.output = self.offset(self.tmp_output, offset_mapping)
        assert len(self.sequence) == len(self.label) == len(self.output)

        # change result format
        self.label = self.change_tag2entity(self.label)
        self.output = self.change_tag2entity(self.output)

        # score
        self.class_info = self.score()

    def argmax(self, output, length):
        # argmax
        output = [torch.argmax(i, dim = -1).numpy().tolist() for i in output]
        # flatten
        new_output = []
        for i in output:
            new_output.extend(i)
        # length
        for i in range(len(new_output)):
            new_output[i] = new_output[i][: length[i]]
        return new_output

    def offset(self, output, offset_mapping):
        new_output = []
        if offset_mapping is None:
            for i in output:
                new_output.append(i[1: -1])
        else:
            offset_mapping = offset_mapping.numpy().tolist()
            for i in range(len(output)):
                item = output[i]
                offs = offset_mapping[i]
                now_output = []
                j = 0
                while(j < len(item)):
                    # remove start and end
                    if j == 0 or j == len(item) - 1:
                        j += 1
                        continue
                    else:
                        all_label_for_one_word = [item[j]]
                        j += 1
                        while(offs[j][0] != 0):
                            all_label_for_one_word.append(item[j])
                            j += 1
                        now_output.append(self.agg_all_label_for_one_word(all_label_for_one_word))
                new_output.append(now_output)
        return new_output

    def agg_all_label_for_one_word(self, label_list):
        for i in label_list:
            if i != 0:
                return i
        return 0

    def change_tag2entity(self, data_y):
        all_entity = []
        for sentence in data_y:
            sentence_entity = []
            sentence = [self.ner_tag.id2tag[i] for i in sentence]
            w = 0
            while(w < len(sentence)):
                if self.ner_tag.ner_tag_method == "BIO":
                    if self.ner_tag.if_tag_first:
                        if sentence[w][0] == "B":
                            now_class = sentence[w].split("-")[1]
                            start = w
                            w = w + 1
                            while(sentence[w] == "I-" + now_class and w < len(sentence)):
                                w += 1
                            sentence_entity.append((now_class, start, w - 1))
                    else:
                        if sentence[w][-1] == "B":
                            now_class = sentence[w].split("-")[0]
                            start = w
                            w = w + 1
                            while(sentence[w] == now_class + "-I" and w < len(sentence)):
                                w += 1
                            sentence_entity.append((now_class, start, w - 1))
                else:
                    raise NotImplementedError(f"please implement the {self.ner_tag.ner_tag_method} method")
            all_entity.append(sentence_entity)
        return all_entity

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def score(self):
        # find rights
        tmp_rights = []
        for s in range(len(self.sequence)):
            right = []
            for w in self.output[s]:
                if w in self.label[s]:
                    right.append(w)
            tmp_rights.append(right)
        
        # extend
        origins = []
        founds = []
        rights = []
        for s in range(len(self.sequence)):
            origins.extend(self.label[s])
            founds.extend(self.output[s])
            rights.extend(tmp_rights[s])

        # 找出每类的个数
        class_info = {}
        origin_counter = Counter([x[0] for x in origins])
        found_counter = Counter([x[0] for x in founds])
        right_counter = Counter([x[0] for x in rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        origin = len(origins)
        found = len(founds)
        right = len(rights)
        recall, precision, f1 = self.compute(origin, found, right)
        class_info["all"] = {"acc": precision, "recall": recall, "f1": f1}
        return class_info

    def get_score(self):
        return self.class_info

    def get_mean_score(self):
        score_dict = {"micro": self.class_info["all"]}
        recall_sum = 0
        precision_sum = 0
        f1_sum = 0
        for i in self.class_info:
            if i != "all":
                recall_sum += self.class_info[i]["recall"]
                precision_sum += self.class_info[i]["acc"]
                f1_sum += self.class_info[i]["f1"]
        score_dict["macro"] = {
            "acc": precision_sum / (len(self.class_info) - 1),
            "recall": recall_sum / (len(self.class_info) - 1),
            "f1": f1_sum / (len(self.class_info) - 1),
        }
        return score_dict
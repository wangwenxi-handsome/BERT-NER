import torch
from collections import Counter


class NERMetric:
    def __init__(self, sequence, entity_labels, entity_outputs):
        self.sequence = sequence
        self.entity_labels = entity_labels
        self.entity_outputs = entity_outputs
        # compute acc, recall, f1
        self.class_info = self.score()

    def _compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def score(self):
        # find rights
        tmp_rights = []
        for s in range(len(self.sequence)):
            right = []
            for w in self.entity_outputs[s]:
                if w in self.entity_labels[s]:
                    right.append(w)
            tmp_rights.append(right)
        
        # extend
        origins = []
        founds = []
        rights = []
        for s in range(len(self.sequence)):
            origins.extend(self.entity_labels[s])
            founds.extend(self.entity_outputs[s])
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
            recall, precision, f1 = self._compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        origin = len(origins)
        found = len(founds)
        right = len(rights)
        recall, precision, f1 = self._compute(origin, found, right)
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
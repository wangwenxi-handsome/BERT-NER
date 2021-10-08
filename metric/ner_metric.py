import torch
from sklearn.metrics import classification_report


class NERMetric:
    def __init__(self, raw_data, output, length, offset_mapping = None):
        self.sequence = raw_data["x"]
        self.label = raw_data["y"]

        # process output
        self.output = self.argmax(output, length.numpy().tolist())
        self.output = self.offset(self.output, offset_mapping.numpy().tolist())
        assert len(self.sequence) == len(self.label) == len(self.output)

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
        for i in range(len(output)):
            item = output[i]
            offs = offset_mapping[i]
            now_output = []
            j = 0
            while(j < range(len(item))):
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

    def show_score(self):
        all_label = []
        all_output = []
        for i in range(len(self.label)):
            all_label.extend(self.label[i])
            all_output.extend(self.output[i])
        print(classification_report(all_label, all_output))

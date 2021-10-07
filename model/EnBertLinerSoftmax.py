import torch.nn as nn
from transformers import AutoModel

class EnBertLinerSoftmax(nn.Module):
    def __init__(self, input_num=768, label_num=15):
        super().__init__()
        self.label_num = label_num
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(0.95)
        self.classifier = nn.Linear(input_num, label_num)

    def forward(self, input_ids, token_type_ids, attention_mask, offset_mapping, label=None):
        output = self.bert(input_ids, token_type_ids, attention_mask)[0]
        output = self.dropout(output)
        output = self.classifier(output)
        return output
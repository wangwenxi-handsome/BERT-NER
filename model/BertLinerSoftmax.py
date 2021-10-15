import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertLinerSoftmax(nn.Module):
    def __init__(self, model_name = "bert-base-cased", label_num=15, dropout_rate=0.95):
        super().__init__()
        self.label_num = label_num
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, label_num)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # cls includes all information
        output = self.bert(input_ids, token_type_ids, attention_mask)[0]
        output = self.dropout(output)
        output = self.classifier(output)
        return output
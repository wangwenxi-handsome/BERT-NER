import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertLinerSoftmax(nn.Module):
    def __init__(
        self, 
        model_name = "bert-base-cased",
        loss_func = "ce",
        label_num=15,
        dropout_rate=0.95,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label_num = label_num
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, label_num)
        self.loss_func = self._get_loss_func(loss_func)

    def forward(self, input_ids, token_type_ids, attention_mask, labels = None):
        # bert returns a special class, the fisrt item is words output.
        output = self.bert(input_ids, token_type_ids, attention_mask)[0]
        output = self.dropout(output)
        output = self.classifier(output)
        loss = None
        if labels is not None:
            # 除去padding后，再计算loss，是否要去除首位占位符还不确定
            # !!!NOTE: to experiment
            active_loss = attention_mask.contiguous().view(-1) == 1
            active_output = output.contiguous().view(-1, self.label_num)[active_loss]
            active_labels = labels.contiguous().view(-1)[active_loss]
            loss = self.loss_func(active_output, active_labels)
        return output, loss

    def _get_loss_func(self, loss_str):
        if loss_str == "ce":
            return nn.CrossEntropyLoss().to(self.device)
import torch.nn as nn

class SentenceNER(nn.Module):
    def __init__(self, input_num=384, label_num=15):
        super(SentenceNER, self).__init__()
        self.label_num = label_num
        self.dropout = nn.Dropout(0.95)
        self.classifier = nn.Linear(input_num, label_num)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.init_weights()

    def forward(self, input, labels=None):
        embedding = self.sentence_transformer.encode(input)
        embedding_output = self.dropout(embedding)
        logits = self.classifier(embedding_output)
        output = (logits, )
        if labels is not None:
            loss = self.loss(logits.contiguous().view(-1, self.label_num), labels.contiguous().view(-1))
            return (logits, loss)
        return output



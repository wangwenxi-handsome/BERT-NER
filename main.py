import torch.optim as optim
import torch.nn as nn

from dataloader import get_ner_dataloader
from model.EnBertLinerSoftmax import EnBertLinerSoftmax
from train.worker import Trainer

dataloader = get_ner_dataloader()
model = EnBertLinerSoftmax()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()
trainer = Trainer(epoch=10, loss_func=loss_func, optimizer=optimizer, dataloader=dataloader, model=model)
trainer.train()
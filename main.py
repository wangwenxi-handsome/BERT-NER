from os import wait
from model.sentence_transformer import SentenceNER
from dataloader.ace05 import wrap_dataloader
from train.base_train import Trainer
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
model = SentenceNER()
train_dataloader, test_dataloader, dev_dataloader = wrap_dataloader()
trainer = Trainer(optimizer, model, train_dataloader, dev_dataloader, test_dataloader)
trainer.train()
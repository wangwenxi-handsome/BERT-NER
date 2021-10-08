import torch.optim as optim
import torch.nn as nn

from dataloader import get_ner_dataloader
from model.EnBertLinerSoftmax import EnBertLinerSoftmax
from train.worker import Trainer
from utils.train import setup_seed

if __name__ == "__main__":
    setup_seed(42)
    dataloader = get_ner_dataloader(batch_size=36)
    model = EnBertLinerSoftmax()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    trainer = Trainer(epoch=50, loss_func=loss_func, optimizer=optimizer, dataloader=dataloader, model=model)
    trainer.train()
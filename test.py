import torch.optim as optim
import torch.nn as nn

from dataloader import get_ner_dataloader
from model.EnBertLinerSoftmax import EnBertLinerSoftmax
from train.worker import Trainer
from utils.train import setup_seed
from metric.ner_metric import NERMetric

if __name__ == "__main__":
    setup_seed(42)
    dataloader, raw_data = get_ner_dataloader(batch_size=36)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    trainer = Trainer(epoch=50, loss_func=loss_func, dataloader=dataloader, load_checkpoint_path = "/Users/bytedance/Desktop/5.pth")
    loss, output = trainer.rollout(trainer.test_dataloader)
    metric = NERMetric(
        raw_data["data_list"]["test"], 
        output, 
        raw_data["data_tensor"]["test"]["length"],
        raw_data["data_tensor"]["test"]["offset_mapping"],
    )
    metric.show_score()
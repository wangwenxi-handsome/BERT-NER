import os
import sys
sys.path.append(os.getcwd())
import torch.optim as optim
import torch.nn as nn

from dataloader.preprocessor.byte_ner import BYTEPreProcessor
from model.BertLinerSoftmax import BertLinerSoftmax
from worker.worker import Worker
from utils.torch_related import setup_seed
from metric.ner_metric import NERMetric

# global args
model_name = "bert-base-chinese"
datapath = "product/data/byte_ner1/raw_data.npy"
label_num = 63
lr = 0.001,
momentum = 0.9,
save_path = "product/data/byte_ner_data_1/checkpoint/"
load_checkpoint_path = None


if __name__ == "__main__":
    # train
    setup_seed(42)
    data_gen = BYTEPreProcessor(datapath = datapath)
    model = BertLinerSoftmax(model_name = model_name, label_num = label_num)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    trainer = Worker(
        loss_func=loss_func, 
        optimizer=optimizer, 
        dataloader=data_gen.get_dataloader(), 
        model=model, 
        save_path=save_path,
        load_checkpoint_path = load_checkpoint_path,
    )
    trainer.train()

    # test
    loss, outputs = trainer.rollout(trainer.test_dataloader)
    metric = NERMetric(
        sequence = data_gen.get_raw_data_x(),
        labels = data_gen.get_raw_data_y(),
        outputs = outputs, 
        tokenize_length = data_gen.get_tokenize_length(),
        ner_tag = data_gen.get_ner_tag(),
    )
    print(metric.get_score())
    print(metric.get_mean_score())    
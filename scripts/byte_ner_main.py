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
data_path = "product/data/byte_ner1/raw_data.pth"
label_num = 63
lr = 0.001
momentum = 0.9
save_checkpoint_path = "product/data/byte_ner1/checkpoint/"
load_checkpoint_path = "product/data/byte_ner1/checkpoint/13.pth"
batch_size = 12
num_workers = 0

if __name__ == "__main__":
    # train
    setup_seed(42)
    data_gen = BYTEPreProcessor(data_path=data_path, model_name=model_name)
    model = BertLinerSoftmax(model_name=model_name, label_num=label_num)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    trainer = Worker(
        loss_func=loss_func, 
        optimizer=optimizer, 
        dataloader=data_gen.get_dataloader(batch_size=batch_size, num_workers=num_workers), 
        model=model, 
        save_checkpoint_path=save_checkpoint_path,
        load_checkpoint_path=load_checkpoint_path,
    )
    # trainer.train()

    # test
    loss, outputs = trainer.rollout(trainer.test_dataloader)
    metric = NERMetric(
        sequence = data_gen.get_raw_data_x("test"),
        labels = data_gen.get_raw_data_y("test"),
        outputs = outputs,
        tokenize_length = data_gen.get_tokenize_length("test"),
        ner_tag = data_gen.get_ner_tag(),
    )
    print(metric.get_score())
    print(metric.get_mean_score())
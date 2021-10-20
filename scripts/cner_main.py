import os
import sys

from torch.utils import data
sys.path.append(os.getcwd())
import torch.optim as optim
import torch.nn as nn

from dataloader.preprocessor.cner import CNERPreProcessor
from model.BertLinerSoftmax import BertLinerSoftmax
from worker.worker import Worker
from utils.torch_related import setup_seed, get_torch_model
from metric.ner_metric import NERMetric

# global args
model_name = "bert-base-chinese"
folder_name = "product/data/cner/data.pth"
label_num = 25
lr = 3e-5
momentum = 0.9
save_checkpoint_path = "product/data/cner/checkpoint"
load_checkpoint_path = None
batch_size = 24
num_workers = 0

if __name__ == "__main__":
    # train
    setup_seed(42)

    # model, opt
    device, model, if_DDP_mode = get_torch_model(
        BertLinerSoftmax, 
        model_config = {"model_name": model_name, "loss_func": "ce", "label_num": label_num},
        load_checkpoint_path = load_checkpoint_path,
    )
    # opt
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)

    # data
    data_gen = CNERPreProcessor(model_name=model_name)
    data_gen.init_data(folder_name=folder_name)
    dataloader = data_gen.get_dataloader(batch_size=batch_size, if_DDP_mode = if_DDP_mode, num_workers=num_workers)

    # worker
    trainer = Worker(
        device = device,
        if_DDP_mode = if_DDP_mode,
        model = model, 
        optimizer = optimizer, 
        save_checkpoint_path = save_checkpoint_path,
    )
    trainer.train(dataloader["train"], dataloader["dev"])

    # test
    outputs, loss = trainer.rollout(dataloader["test"])
    entity_outputs, entity_labels = data_gen.decode(
        outputs, 
        data_gen.get_tokenize_length("test"), 
        data_gen.get_raw_data_y("test"),
    )
    # metric
    metric = NERMetric(data_gen.get_raw_data_x("test"), entity_labels, entity_outputs)
    print(metric.get_score())
    print(metric.get_mean_score())
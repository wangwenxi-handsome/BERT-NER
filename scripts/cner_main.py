import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.optim as optim

from dataloader.preprocessor.cner import CNERPreProcessor
from model.BertLinerSoftmax import BertLinerSoftmax
from worker.worker import Worker
from utils.torch_related import setup_seed, get_torch_model
from metric.ner_metric import NERMetric
from utils.torch_related import get_linear_schedule_with_warmup


# global args
epoch = 3
model_name = "bert-base-chinese"
folder_name = "/opt/tiger/data.pth"
label_num = 25
lr = 3e-05
save_checkpoint_path = "product/data/cner/checkpoint"
load_checkpoint_path = None
batch_size_per_gpu = 24
num_workers = 0


if __name__ == "__main__":
    # train
    setup_seed(42)

    # model, opt
    device, model = get_torch_model(
        BertLinerSoftmax, 
        model_config = {"model_name": model_name, "loss_func": "ce", "label_num": label_num},
        load_checkpoint_path = load_checkpoint_path,
    )
    # opt
    # optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr = lr, eps = 1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 48, num_training_steps = 480)

    # data
    data_gen = CNERPreProcessor(model_name=model_name)
    data_gen.init_data(folder_name=folder_name)
    n_gpus = torch.cuda.device_count() if device != "cpu" else 1
    dataloader = data_gen.get_dataloader(batch_size=batch_size_per_gpu * n_gpus, num_workers=num_workers)

    # worker
    trainer = Worker(
        epoch = epoch,
        device = device,
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler,
        save_checkpoint_path = save_checkpoint_path,
        epoch = epoch,
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
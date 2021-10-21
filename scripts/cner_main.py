import os
import sys
sys.path.append(os.getcwd())
import torch.optim as optim

from dataloader.preprocessor.cner import CNERPreProcessor
from model.BertLinerSoftmax import BertLinerSoftmax
from worker.worker import Worker
from utils.torch_related import setup_seed, get_torch_model
from metric.ner_metric import NERMetric


# global args
epoch = 5
model_name = "bert-base-chinese"
folder_name = "/opt/tiger/data.pth"
label_num = 25
lr = 0.001
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
    # !!!NOTE I can not understand
    # optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=3e-05, eps=1e-08)
    from utils.torch_related import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=48, num_training_steps=480)

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
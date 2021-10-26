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
model_name = "/opt/tiger/bert-base-chinese"
folder_name = "product/data/cner"
label_num = 25
lr = 3e-05
save_checkpoint_path = "product/data/cner/checkpoint"
# load_checkpoint_path = "product/data/cner/checkpoint/1.pth"
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 24, num_training_steps = 240.0)

    # data
    data_gen = CNERPreProcessor(model_name=model_name)
    data_gen.init_data(folder_name=folder_name)
    n_gpus = max(torch.cuda.device_count(), 1)
    dataloader = data_gen.get_dataloader(batch_size=batch_size_per_gpu * n_gpus, num_workers=num_workers)

    # worker
    trainer = Worker(
        epoch = epoch,
        device = device,
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler,
        save_checkpoint_path = save_checkpoint_path,
    )
    trainer.train(dataloader["train"], dataloader["dev"])

    # test metric
    outputs, loss = trainer.rollout(dataloader["dev"])
    entity_outputs, entity_labels, offset_outputs = data_gen.decode(
        outputs, 
        data_gen.get_tokenize_length("dev"), 
        data_gen.get_raw_data_y("dev"),
    )
    metric = NERMetric(data_gen.get_raw_data_x("dev"), entity_labels, entity_outputs)
    print(metric.get_score())
    print(metric.get_mean_score())
    
    # find loc error
    outputs_extend = []
    for i in outputs:
        outputs_extend.extend(i)
    print(data_gen.get_ner_tag().tag2id)
    print(data_gen.get_ner_tag().id2tag)
    data_x = data_gen.get_raw_data_x("dev")
    data_y = data_gen.get_raw_data_y("dev")
    num = 0
    for i in range(len(data_x)):
        if 8 in data_y[i] or 9 in data_y[i]:
            if num <= 10:
                print(data_x[i])
                print(data_y[i])
                print(offset_outputs[i])
                num += 1
    print(num)
    
        
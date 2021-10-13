import os
import sys
sys.path.append(os.getcwd())
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizerFast

from dataloader import get_ner_dataloader
from dataloader.dataset.byte_ner import BYTENER
from dataloader.processor.ner_processor import NERProcessor, NERDataSet
from model.EnBertLinerSoftmax import BertLinerSoftmax
from train.worker import Trainer
from utils.train import setup_seed
from utils.data import dict_to_list_by_max_len
from metric.ner_metric import NERMetric

# global args
model_name = "bert-base-chinese"
random_seed = 42

# kwargs config
config = {
    "dataloader": {
        "data_cls": BYTENER,
        "data_config": {
            "ner_tag_method": "BIO",
            "base_config": {
                "task": "NER",
                "folder_path": "./product/data/byte_ner_data_1",
                "train_fn": "raw_data.npy",
                "test_fn": None,
                "dev_fn": None,
                "cased": True,
                "if_tag_first": True,
                "test_rate": 0.1,
                "dev_rate": 0.1,
                "cross_valid": False,
                "random_state": random_seed,
            }
        },
        "processor_cls": NERProcessor,
        "processor_config": {
            "model_name": model_name,
            "tokenizer_cls": BertTokenizerFast,
            "language": "cn",
            "is_split_into_words": True,
            "return_offsets_mapping": False,
            "padding": "max_length",
            "truncation": True,
            "max_length": None,
            "return_tensors": "pt",
        },
        "dataset_cls": NERDataSet,
        "batch_size": 24,
        "num_workers": 0,
        "collate_fn": dict_to_list_by_max_len,
        "raw_data": True,
    },

    "model": {
        "model_name": model_name,
        "label_num": 63,
        "dropout_rate": 0.95,
    },

    "optim": {
        "lr": 0.001,
        "momentum": 0.9,
    },

    "train": {
        "epoch": 50,
        "load_checkpoint_path": None,
        "if_by_state_dict": False,
        "save_path": "./product/model",
    },
}

if __name__ == "__main__":
    # train
    setup_seed(random_seed)
    dataloader, raw_data = get_ner_dataloader(**config["dataloader"])
    model = BertLinerSoftmax(**config["model"])
    optimizer = optim.SGD(model.parameters(), **config["optim"])
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    trainer = Trainer(loss_func=loss_func, optimizer=optimizer, dataloader=dataloader, model=model, **config["train"])
    trainer.train()

    # test
    loss, output = trainer.rollout(trainer.test_dataloader)
    metric = NERMetric(
        sequence = raw_data["data_list"]["test"]["x"],
        label = raw_data["data_list"]["test"]["y"],
        output = output, 
        length = raw_data["data_tensor"]["test"]["length"],
        ner_tag = raw_data["ner_tag"]
    )
    print(metric.get_score())
    print(metric.get_mean_score())
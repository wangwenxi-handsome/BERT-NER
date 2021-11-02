import os
import sys
sys.path.append(os.getcwd())
from dataloader.preprocessor.cner import CNERPreProcessor
from scripts.run_ner import run_ner


global_config = {
    # path
    "data_cls": CNERPreProcessor,
    "data_folder_name": "product/data/cner/data.pth",
    "folder_path": "product/experiments/cner1/",
    # model
    "model_name": "bert-base-chinese",
    "label_num": 25,
    # train
    "epoch": 3,
    "lr": 3e-05,
    "batch_size_per_gpu": 24,
    "save_step_rate": 0.1,
}


if __name__ == "__main__":
    run_ner(global_config)
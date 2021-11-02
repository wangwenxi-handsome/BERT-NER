import os
import sys
sys.path.append(os.getcwd())
from dataloader.preprocessor.byte_ner import BYTEPreProcessor
from scripts.run_ner import run_ner


global_config = {
    # path
    "data_cls": BYTEPreProcessor,
    "data_folder_name": "product/data/byte_ner1/con_unchecked1019.npy",
    "folder_path": "product/experiments/byte1/",
    # model
    "model_name": "bert-base-chinese",
    "label_num": 63,
    # train
    "epoch": 5,
    "lr": 5e-05,
    "batch_size_per_gpu": 24,
    "save_step_rate": 0.1,
}


if __name__ == "__main__":
    run_ner(global_config)
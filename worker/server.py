import os
import time
import torch.nn as nn
import torch.optim as optim

from dataloader.preprocessor.byte_ner import BYTETESTPreProcessor, BYTETrainPreProcessor
from worker.worker import Worker
from utils.torch_related import setup_seed


class BYTENERServer:
    def __init__(
        self,
        model_name,
        load_checkpoint_path,
        batch_size = 12,
    ):
        # kwargs
        self.model_name = model_name
        self.load_checkpoint_path = load_checkpoint_path
        self.batch_size = batch_size

        # prepare model
        setup_seed(42)
        self.trainer = Worker(load_checkpoint_path=load_checkpoint_path)

    def predict(self, data):
        data_gen = BYTETESTPreProcessor(data_path=data, model_name=self.model_name)
        _, outputs = self.trainer.rollout(data_gen.get_dataloader(batch_size=self.batch_size)["test"])
        entity_outputs, _ = data_gen.decode(
            outputs, 
            data_gen.get_tokenize_length("test"), 
        )
        return dict(zip(data_gen.get_raw_data_id("test"), entity_outputs))

    def train(self, data, lr = 0.001, momentum = 0.9):
        data_gen = BYTETrainPreProcessor(
            data_path=data, 
            model_name=self.model_name,
        )
        # 设置worker的训练参数
        new_dataloader = data_gen.get_dataloader(batch_size=self.batch_size)
        self.trainer.updata_train_kwargs(
            os.path.dirname(self.load_checkpoint_path),
            new_dataloader["train"],
            new_dataloader["dev"],
            optim.SGD(self.trainer.model.parameters(), lr=lr, momentum=momentum),
            nn.CrossEntropyLoss(ignore_index=0),
        )
        # 训练并更新模型
        self.trainer.train()
        self.trainer.model = self.trainer.best_model.to(self.trainer.device)
        print("model update")
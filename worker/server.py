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

        # prepare data
        self.predict_data_gen = BYTETESTPreProcessor(model_name=self.model_name)
        self.train_data_gen = BYTETrainPreProcessor(model_name=self.model_name)

        # prepare model
        setup_seed(42)
        self.trainer = Worker(load_checkpoint_path=load_checkpoint_path)

    def predict(self, data):
        self.predict_data_gen.init_data(data)
        _, outputs = self.trainer.rollout(self.predict_data_gen.get_dataloader(batch_size=self.batch_size)["test"])
        entity_outputs, _ = self.predict_data_gen.decode(
            outputs, 
            self.predict_data_gen.get_tokenize_length("test"), 
        )
        return dict(zip(self.predict_data_gen.get_raw_data_id("test"), entity_outputs))

    def train(self, data, lr = 0.001, momentum = 0.9):
        self.train_data_gen.init_data(data)
        # 设置worker的训练参数
        new_dataloader = self.train_data_gen.get_dataloader(batch_size=self.batch_size)
        self.trainer.update_train_kwargs(
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
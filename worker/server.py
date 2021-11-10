import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

from metric.ner_metric import NERMetric
from model.BertLinerSoftmax import BertLinerSoftmax
from dataloader.preprocessor.byte_ner import BYTEServingPreProcessor
from worker.worker import Worker
from utils.torch_related import setup_seed, get_torch_model, get_linear_schedule_with_warmup


class BYTENERServer:
    def __init__(
        self,
        model_name,
        load_checkpoint_path,
        batch_size = 12,
        label_num = 63,
    ):
        setup_seed(42)
        # kwargs
        self.model_name = model_name
        self.load_checkpoint_path = load_checkpoint_path
        self.batch_size = batch_size
        self.label_num = label_num

        # data gen
        self.train_data_gen = BYTEServingPreProcessor(
            model_name = self.model_name,
            dataloader_name = ["train", "dev"],
            split_rate = [0.1],
            ner_tag_method = "BIO",
            max_length = 512,
        )
        self.predict_data_gen = BYTEServingPreProcessor(
            model_name = self.model_name,
            dataloader_name = ["test"],
            split_rate = [],
            ner_tag_method = "BIO",
            max_length = 512,
        )

        # model
        self.n_gpus = max(torch.cuda.device_count(), 1)
        self.device, self.model = get_torch_model(
            BertLinerSoftmax, 
            model_config = {"model_name": self.model_name, "loss_func": "ce", "label_num": self.label_num},
            load_checkpoint_path = load_checkpoint_path,
            if_by_state_dict = True,
        )

    def predict(self, data):
        self.predict_data_gen.init_data(data)
        worker = Worker(device = self.device, model = self.model)
        _, outputs = worker.rollout(self.predict_data_gen.get_dataloader(batch_size = self.batch_size * self.n_gpus)["test"])
        entity_outputs, entity_labels, _ = self.predict_data_gen.decode(
                outputs, 
                self.predict_data_gen.get_tokenize_length("test"), 
                self.predict_data_gen.get_raw_data_y("test"),
            )
        return dict(zip(self.predict_data_gen.get_raw_data_id("test"), entity_outputs))

    def train(self, data, folder_name, epoch = 3, lr = 5e-05):
        # data
        self.train_data_gen.init_data(data)
        dataloader = self.train_data_gen.get_dataloader(batch_size=self.batch_size)

        # train and select
        # train
        # opt
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        sum_steps = epoch * len(dataloader["train"])
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr = lr, eps = 1e-08)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.1 * sum_steps, num_training_steps = sum_steps)

        # worker
        trainer = Worker(
            device = self.device,
            model = self.model, 
            folder_path = folder_name,
            epoch = epoch,
            optimizer = optimizer, 
            scheduler = scheduler,
            save_step_rate = 0.1,
        )
        trainer.train(dataloader["train"])

        # select best model
        metrics = []
    
        all_checkpoints = os.listdir(os.path.join(folder_name, "model/"))
        if ".ipynb_checkpoints" in all_checkpoints:
            all_checkpoints.remove(".ipynb_checkpoints")   
        if "best_model.pth" in all_checkpoints:
            all_checkpoints.remove("best_model.pth")  
        all_checkpoints.sort(key = lambda x: int(x.split(".")[0]))
        all_checkpoints = [os.path.join(folder_name, "model/", i) for i in all_checkpoints]
        
        best_checkpoint = None
        best_micro_f1 = None
        
        for checkpoint in all_checkpoints:
            device, model = get_torch_model(
                BertLinerSoftmax, 
                model_config = {"model_name": self.model_name, "loss_func": "ce", "label_num": self.label_num},
                load_checkpoint_path = checkpoint,
            )

            trainer = Worker(
                device = device,
                model = model, 
            )

            outputs, _ = trainer.rollout(dataloader)
            entity_outputs, entity_labels, _ = self.train_data_gen.decode(
                outputs, 
                self.train_data_gen.get_tokenize_length("dev"), 
                self.train_data_gen.get_raw_data_y("dev"),
            )
            metric = NERMetric(self.train_data_gen.get_raw_data_x("dev"), entity_labels, entity_outputs)
            metrics.append(metric)
            
            # update best_checkpoint
            if best_checkpoint is None or metric.get_mean_score()["micro"]["f1"] > best_micro_f1:
                best_checkpoint = checkpoint
                best_micro_f1 = metric.get_mean_score()["micro"]["f1"]
        
        # select best model
        save_best_model = os.path.join(folder_name, "model/best_model.pth")
        shutil.copyfile(best_checkpoint, save_best_model)
        _, self.model = get_torch_model(
                BertLinerSoftmax, 
                model_config = {"model_name": self.model_name, "loss_func": "ce", "label_num": self.label_num},
                load_checkpoint_path = save_best_model,
        )
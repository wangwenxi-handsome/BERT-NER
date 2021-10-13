import os
from typing import Callable
from tqdm import tqdm
import torch
import torch.nn as nn


class Worker:
    def __init__(
        self, 
        loss_func: Callable,
        model: nn.Module = None,
        load_checkpoint_path: str = None,
        if_by_state_dict: bool = False,
        save_path = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.if_by_state_dict = if_by_state_dict
        self.loss_func = loss_func
        self.save_path = save_path
        self.load_model(model, load_checkpoint_path, if_by_state_dict)

    def load_model(self, model, load_checkpoint_path, if_by_state_dict):
        # init model
        if load_checkpoint_path is None:
            self.model = model
        # load model from state_dict
        elif if_by_state_dict:
            self.model = model
            self.model.load_state_dict(torch.load(load_checkpoint_path, map_location=self.device))
        # load model from pth
        else:
            self.model = torch.load(load_checkpoint_path)

    def save_model(self, save_path):
        if self.if_by_state_dict:
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model, save_path)


class Trainer(Worker):
    def __init__(
        self, 
        loss_func,
        dataloader,
        epoch = 50,
        optimizer = None, 
        model: nn.Module = None,
        load_checkpoint_path: str = None,
        if_by_state_dict: bool = False,
        save_path = "./product/model",
    ):
        super().__init__(loss_func, model, load_checkpoint_path, if_by_state_dict, save_path)
        self.epoch = epoch
        self.opt = optimizer
        self.train_dataloader = dataloader.get("train", None)
        self.dev_dataloader = dataloader.get("dev", None)
        self.test_dataloader = dataloader.get("test", None)

        # early stop
        self.best_loss = None
        self.best_loss_epoch = None

    def train(self):
        self.model.to(self.device)
        for e in range(self.epoch):
            print(f"#############epoch{e + 1}#############")
            step = 0
            accum_loss = 0
            for data in tqdm(self.train_dataloader):
                # zero grad
                self.opt.zero_grad()

                # model forward
                data = {i: data[i].to(self.device) for i in data}
                label = data["label"]
                output = self.model(**data)

                # get loss and step
                loss = self.loss_func(output.contiguous().view(-1, self.model.label_num), label.contiguous().view(-1))
                loss.backward()
                self.opt.step()

                # valid
                step += 1
                accum_loss += loss
                if step % 50 == 0:
                    print(f"train loss is {accum_loss / 50}")
                    step = 0
                    accum_loss = 0

            # valid
            valid_loss, _ = self.rollout(self.dev_dataloader)
            if self.best_loss is None or valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_epoch = e
                # save model
                if e > 3:
                    self.save_model(os.path.join(self.save_path, f"{e + 1}.pth"))
            elif e - self.best_loss_epoch > 2:
                break

    def rollout(self, dataloader):
        outputs = []
        with torch.no_grad():
            self.model.eval()
            loss = 0
            for data in dataloader:
                data = {i: data[i].to(self.device) for i in data}
                label = data["label"]
                output = self.model(**data)
                outputs.append(output.cpu())
                loss += self.loss_func(output.contiguous().view(-1, self.model.label_num), label.contiguous().view(-1))
            loss /= len(dataloader)
            print(f"valid loss is {loss}")
            self.model.train()
            return loss, outputs
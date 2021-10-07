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
        load_by_state_dict: bool = True,
    ):
        self.load_model(model, load_checkpoint_path, load_by_state_dict)
        self.save_by_state_dict = load_by_state_dict
        self.loss_func = loss_func
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self, model, load_checkpoint_path, load_by_state_dict):
        if load_checkpoint_path is None:
            self.model = model
        elif load_by_state_dict:
            self.model = model
            self.model.load_state_dict(torch.load(load_checkpoint_path))
        else:
            self.model = torch.load(load_checkpoint_path)

    def save_model(self, save_path, save_by_state_dict=True):
        if save_by_state_dict:
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model, save_path)


class Trainer(Worker):
    def __init__(
        self, 
        epoch,
        loss_func, 
        optimizer, 
        dataloader,
        model: nn.Module = None,
        load_checkpoint_path: str = None,
        load_by_state_dict: bool = True,
    ):
        super().__init__(loss_func, model, load_checkpoint_path, load_by_state_dict)
        self.epoch = epoch
        self.opt = optimizer
        self.train_dataloader = dataloader["train"]
        self.dev_dataloader = dataloader["dev"]
        self.test_dataloader = dataloader["test"]

        # early stop
        self.best_loss = None
        self.best_loss_epoch = None

    def train(self):
        self.model.to(self.device)
        for e in tqdm(range(self.epoch)):
            print(f"#############epoch{e + 1}#############")
            step = 0
            accum_loss = 0
            for data in self.train_dataloader:
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
                if step % 100 == 0:
                    print(f"train loss is {accum_loss / 100}")
                    step = 0
                    accum_loss = 0

            # valid
            valid_loss = self.rollout(self.dev_dataloader)
            if self.best_loss is None or valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_epoch = e
            elif e - self.best_loss_epoch > 2:
                break

    def rollout(self, dataloader):
        loss = 0
        for data in dataloader:
            data = {i: data[i].to(self.device) for i in data}
            label = data["label"]
            output = self.model(**data)
            loss += self.loss_func(output.contiguous().view(-1, self.model.label_num), label.contiguous().view(-1))
        loss /= len(dataloader)
        return loss
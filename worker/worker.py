import os
import copy
import inspect
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter   


class Worker:
    """train and rollout.
    """
    def __init__(
        self, 
        model: nn.Module = None,
        dataloader = {},
        optimizer = None, 
        if_by_state_dict: bool = False,
        load_checkpoint_path: str = None,
        save_checkpoint_path: str = None,
        epoch = 50,
    ):
        # train kwargs
        self.epoch = epoch

        # data
        self.train_dataloader = dataloader.get("train", None)
        self.dev_dataloader = dataloader.get("dev", None)
        self.test_dataloader = dataloader.get("test", None)

        # torch related. model, opt with device.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.opt = optimizer
        self.if_by_state_dict = if_by_state_dict
        self.save_checkpoint_path = save_checkpoint_path
        self.load_model(model, load_checkpoint_path, if_by_state_dict)

        # early stop
        self.best_loss = None
        self.best_loss_epoch = None
        self.best_model = None

        # tensorboard
        self.writer = SummaryWriter('product/log/')

    def train(self):
        total_step = 0
        for e in range(self.epoch):
            print(f"This is epoch{e}#")
            step = 0
            accum_loss = 0
            for data in tqdm(self.train_dataloader):
                # zero grad
                self.opt.zero_grad()

                # model forward
                model_kwargs = dict(inspect.signature(self.model.forward).parameters)
                model_input = {}
                for i in model_kwargs:
                    tmp_input = data.get(i, None)
                    if tmp_input is not None:
                        model_input[i] = tmp_input.to(self.device)
                    else:
                        model_input[i] = None
                output, loss = self.model(**model_input)

                # step
                loss.backward()
                self.opt.step()

                # print loss every 1/5
                step += 1
                accum_loss += loss
                total_step += 1
                self.writer.add_scalar("loss", loss, total_step)
                if step % int(len(self.train_dataloader) / 5) == 0:
                    temp_loss = accum_loss / int(len(self.train_dataloader) / 5)
                    print(f"train loss is {temp_loss}")
                    step = 0
                    accum_loss = 0

            # valid
            outputs, valid_loss = self.rollout(self.dev_dataloader)
            # if best model
            if self.best_loss is None or valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_epoch = e
                self.best_model = copy.deepcopy(self.model).cpu()
            elif e - self.best_loss_epoch > 2:
                if self.save_checkpoint_path is not None:
                    # save model
                    if not os.path.exists(self.save_checkpoint_path):
                        os.mkdir(self.save_checkpoint_path)
                    self.save_model(self.best_model, os.path.join(self.save_checkpoint_path, f"{self.best_loss_epoch}.pth"))
                break

    def rollout(self, dataloader):
        outputs = []
        # model forward to get outputs
        with torch.no_grad():
            self.model.eval()
            loss_sum = None
            for data in dataloader:
                # model forward
                model_kwargs = dict(inspect.signature(self.model.forward).parameters)
                model_input = {}
                for i in model_kwargs:
                    tmp_input = data.get(i, None)
                    if tmp_input is not None:
                        model_input[i] = tmp_input.to(self.device)
                    else:
                        model_input[i] = None
                output, loss = self.model(**model_input)
                outputs.append(output.cpu())
                if loss is not None:
                    if loss_sum is None:
                        loss_sum = loss
                    else:
                        loss_sum += loss
            self.model.train()
        
        # return outputs and loss(None is no labels)
        if loss_sum is not None:
            loss_mean = loss_sum / len(dataloader)
            print(f"valid loss is {loss_mean}")
        return outputs, loss_mean

    def load_model(self, model, load_checkpoint_path, if_by_state_dict):
        """load_checkpoint_path have high prority.
        """
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
        self.model.to(self.device)

    def save_model(self, model, save_path):
        # if_by_state_dict applied to save and load means 从哪里来就到哪里去
        if self.if_by_state_dict:
            torch.save(model.state_dict(), save_path)
        else:
            torch.save(model, save_path)

    def update_train_kwargs(
        self,
        save_checkpoint_path = None, 
        train_dataloader = None, 
        dev_dataloader = None,
        opt = None,
    ):
        if save_checkpoint_path is not None:
            self.save_checkpoint_path = save_checkpoint_path
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if dev_dataloader is not None:
            self.dev_dataloader = dev_dataloader
        if opt is not None:
            self.opt = opt
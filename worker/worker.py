import os
import copy
import inspect
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter   


class Worker:
    """train and rollout.
    """
    def __init__(
        self, 
        device,
        if_DPP_mode,
        model: nn.Module,
        optimizer = None, 
        scheduler = None,
        save_checkpoint_path: str = None,
        if_by_state_dict: bool = False,
        epoch = 50,
    ):
        # train kwargs
        self.epoch = epoch

        # device
        self.device = device
        self.if_DPP_mode = if_DPP_mode

        # torch related. model, opt with device.
        self.opt = optimizer
        self.scheduler = self.scheduler
        self.if_by_state_dict = if_by_state_dict
        self.save_checkpoint_path = save_checkpoint_path
        self.model = model

        # early stop
        self.best_loss = None
        self.best_loss_epoch = None
        self.best_model = None

        # tensorboard
        self.writer = SummaryWriter('product/log/')

    def train(self, train_dataloader, dev_dataloader):
        total_step = 0
        for e in range(self.epoch):
            print(f"This is epoch{e}#")
            step = 0
            accum_loss = 0
            # DDP：设置sampler的epoch，
            # DistributedSampler需要这个来指定shuffle方式，
            # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
            if self.if_DPP_mode:
                train_dataloader.sampler.set_epoch(e)
                
            for data in tqdm(train_dataloader):
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
                if self.scheduler is not None:
                    self.scheduler.step()

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
            outputs, valid_loss = self.rollout(dev_dataloader)
            # if best model
            if self.best_loss is None or valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_epoch = e
                if self.if_DPP_mode:
                    if dist.get_rank() == 0:
                        self.best_model = copy.deepcopy(self.model.module).cpu()
                else:
                    self.best_model = copy.deepcopy(self.model).cpu()
            # early stop
            elif e - self.best_loss_epoch > 2:
                if self.save_checkpoint_path is not None:
                    # save model
                    if (self.if_DPP_mode and dist.get_rank() == 0) or (not self.if_DPP_mode):
                        if not os.path.exists(self.save_checkpoint_path):
                            os.mkdir(self.save_checkpoint_path)
                        self.save_model(self.best_model, os.path.join(self.save_checkpoint_path, f"{self.best_loss_epoch}.pth"))
                break

    @torch.no_grad()
    def rollout(self, dataloader):
        outputs = []

        # model forward to get outputs
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
        loss_mean = None
        if loss_sum is not None:
            loss_mean = loss_sum / len(dataloader)
            print(f"valid loss is {loss_mean}")
        return outputs, loss_mean

    def save_model(self, model, save_path):
        # if_by_state_dict applied to save and load means 从哪里来就到哪里去
        if self.if_by_state_dict:
            torch.save(model.state_dict(), save_path)
        else:
            torch.save(model, save_path)

    def update_train_kwargs(
        self,
        save_checkpoint_path = None, 
        opt = None,
    ):
        """补全或更新训练时缺失的参数
        """
        if save_checkpoint_path is not None:
            self.save_checkpoint_path = save_checkpoint_path
        if opt is not None:
            self.opt = opt
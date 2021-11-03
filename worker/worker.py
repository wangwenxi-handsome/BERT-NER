import os
import copy
import inspect
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.progressbar import ProgressBar


class Worker:
    """train and rollout.
    """
    def __init__(
        self, 
        device,
        model: nn.Module,
        epoch = 10,
        folder_path = None,
        optimizer = None, 
        scheduler = None,
        if_by_state_dict: bool = True,
        save_step_rate = 0.1,
    ):
        # train and rollout
        self.device = device
        self.model = model

        # only for train
        # train kwargs
        self.epoch = epoch

        # torch related. model, opt with device.
        self.opt = optimizer
        self.scheduler = scheduler

        # early stop
        self.best_loss = None
        self.best_loss_epoch = None
        self.best_model = None

        # save model and tensorboard writer
        if folder_path is not None:
            self.log_path = os.path.join(folder_path, "log/")
            self.save_model_path = os.path.join(folder_path, "model/")
            if not os.path.exists(self.log_path):
                    os.mkdir(self.log_path)
            if not os.path.exists(self.save_model_path):
                    os.mkdir(self.save_model_path)

            self.if_by_state_dict = if_by_state_dict
            self.save_step_rate = save_step_rate
            self.writer = SummaryWriter(self.log_path)

    def train(self, train_dataloader, dev_dataloader = None):
        global_step = 0
        save_step = int(self.epoch * len(train_dataloader) * self.save_step_rate)
        # custom progress par
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(self.epoch))
        for e in range(self.epoch):
            pbar.reset()
            pbar.epoch_start(current_epoch = e)
            step = 0
            # DistributedSampler需要这个来指定shuffle方式，
            # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果
            if dist.is_initialized():
                train_dataloader.sampler.set_epoch(e)
            
            for step, data in enumerate(train_dataloader):
                # zero grad
                self.opt.zero_grad()

                # model forward
                if hasattr(self.model, "module"):
                    func = self.model.module.forward
                else:
                    func = self.model.forward
                model_kwargs = dict(inspect.signature(func).parameters)
                model_input = {}
                for i in model_kwargs:
                    tmp_input = data.get(i, None)
                    if tmp_input is not None:
                        model_input[i] = tmp_input.to(self.device)
                    else:
                        model_input[i] = None
                output, loss = self.model(**model_input)
                if len(loss) > 1:
                    loss = loss.mean()

                # step
                # 反向传播计算梯度
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # 参数更新
                self.opt.step()
                # 学习率更新
                if self.scheduler is not None:
                    self.scheduler.step()

                # save
                if (global_step + 1) % save_step == 0:
                    self.save_model(os.path.join(self.save_model_path, f"{global_step}.pth"))

                # bar progress and tensorboard writer
                pbar(step, {'loss': loss.item()})
                self.writer.add_scalar("loss", loss, global_step)
                global_step += 1

            # dev
            if dev_dataloader is not None:
                _, valid_loss = self.rollout(dev_dataloader)

                # early stop
                # 早停只用于判断训练到这个epoch时已经不可能是最佳模型了，而不是判断哪个模型最佳的
                # 判断最佳模型的方法是，每隔固定step保存模型，然后训练结束后对所有模型在验证集上计算指标，作图选取
                if self.best_loss is None or valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.best_loss_epoch = e
                elif e - self.best_loss_epoch > 2:
                    break

    @torch.no_grad()
    def rollout(self, dataloader):
        outputs = []

        # model forward to get outputs
        self.model.eval()
        loss_sum = None
        pbar = ProgressBar(n_total=len(dataloader), desc="Rollout")
        for step, data in enumerate(dataloader):
            pbar.reset()

            # model forward
            if hasattr(self.model, "module"):
                func = self.model.module.forward
            else:
                func = self.model.forward
            model_kwargs = dict(inspect.signature(func).parameters)
            model_input = {}
            for i in model_kwargs:
                tmp_input = data.get(i, None)
                if tmp_input is not None:
                    model_input[i] = tmp_input.to(self.device)
                else:
                    model_input[i] = None
            output, loss = self.model(**model_input)
            if len(loss) > 1:
                loss = loss.mean()
            outputs.append(output.cpu())
            if loss is not None:
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
            pbar(step)
        self.model.train()
        
        # return outputs and loss(None is no labels)
        loss_mean = None
        if loss_sum is not None:
            loss_mean = loss_sum / len(dataloader)
            print(f"valid loss is {loss_mean}")
        return outputs, loss_mean

    def save_model(self, save_path):
        if hasattr(self.model, "module"):
            save_model = copy.deepcopy(self.model.module).cpu()
        else:
            save_model = copy.deepcopy(self.model).cpu()

        if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
            if self.if_by_state_dict:
                torch.save(save_model.state_dict(), save_path)
            else:
                torch.save(save_model, save_path)
import os
import shutil
import torch
import torch.optim as optim
from model.BertLinerSoftmax import BertLinerSoftmax
from worker.worker import Worker
from utils.torch_related import setup_seed, get_torch_model
from metric.ner_metric import NERMetric
from utils.torch_related import get_linear_schedule_with_warmup
from utils.logger import init_logger


# defalut kwargs for NER
defalut_config = {
    # path
    "data_cls": None,   # 模型训练用到的预处理类
    "data_folder_name": "product/data/cner/data.pth",   # 原始数据位置 
    "folder_path": "product/experiments/cner1/",    # 本次实验log，checkpoint的保存位置
    # model
    "model_name": "bert-base-chinese",  # 用于选择使用哪一款bert模型
    "label_num": 25,    # NER标签的数目
    # train
    "epoch": 3,     # epoch数
    "lr": 2e-05,    # 学习率
    "batch_size_per_gpu": 24,   # 每张显卡上的batch_size
    "save_step_rate": 0.1,  # 每训练多少百分比保存一个checkpoint
    # main
    "if_train": True,
    "if_select": True,
    "if_test": True,
}


def train(logger, config, data_gen, train_dataloader, dev_dataloader = None):
    # model
    device, model = get_torch_model(
        BertLinerSoftmax, 
        model_config = {"model_name": config["model_name"], "loss_func": "ce", "label_num": config["label_num"]},
        load_checkpoint_path = None,
    )

    # opt
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    sum_steps = config["epoch"] * len(train_dataloader)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr = config["lr"], eps = 1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.1 * sum_steps, num_training_steps = sum_steps)

    # worker
    trainer = Worker(
        device = device,
        model = model, 
        folder_path = config["folder_path"],
        epoch = config["epoch"],
        optimizer = optimizer, 
        scheduler = scheduler,
        save_step_rate = config["save_step_rate"],
    )
    trainer.train(train_dataloader, dev_dataloader)


def select(logger, config, data_gen, dataloader, name):
    metrics = []
    
    all_checkpoints = os.listdir(os.path.join(config["folder_path"], "model/"))
    if ".ipynb_checkpoints" in all_checkpoints:
        all_checkpoints.remove(".ipynb_checkpoints")   
    if "best_model.pth" in all_checkpoints:
        all_checkpoints.remove("best_model.pth")  
    all_checkpoints.sort(key = lambda x: int(x.split(".")[0]))
    all_checkpoints = [os.path.join(config["folder_path"], "model/", i) for i in all_checkpoints]
    logger.info("all checkpoints %s", all_checkpoints)
    
    best_checkpoint = None
    best_micro_f1 = None
    
    for checkpoint in all_checkpoints:
        logger.info("select by %s start with %s", name, checkpoint)
        device, model = get_torch_model(
            BertLinerSoftmax, 
            model_config = {"model_name": config["model_name"], "loss_func": "ce", "label_num": config["label_num"]},
            load_checkpoint_path = checkpoint,
        )

        trainer = Worker(
            device = device,
            model = model, 
        )

        outputs, _ = trainer.rollout(dataloader)
        entity_outputs, entity_labels, offset_outputs = data_gen.decode(
            outputs, 
            data_gen.get_tokenize_length(name), 
            data_gen.get_raw_data_y(name),
        )
        metric = NERMetric(data_gen.get_raw_data_x(name), entity_labels, entity_outputs)
        metrics.append(metric)
        logger.info("metric is %s", metric.get_score())
        logger.info("mean metric is %s", metric.get_mean_score())
        
        # update best_checkpoint
        if best_checkpoint is None or metric.get_mean_score()["micro"]["f1"] > best_micro_f1:
            best_checkpoint = checkpoint
            best_micro_f1 = metric.get_mean_score()["micro"]["f1"]
    
    # select best model
    save_best_model = os.path.join(config["folder_path"], "model/best_model.pth")
    shutil.copyfile(best_checkpoint, save_best_model)

    # plot
    acc = []
    recall = []
    f1 = []
    for m in metrics:
        mean_score = m.get_mean_score()
        acc.append(mean_score["micro"]["acc"])
        recall.append(mean_score["micro"]["recall"])
        f1.append(mean_score["micro"]["f1"])
    
    """
    l1 = plt.plot(acc,'r--', label='acc')
    l2 = plt.plot(recall,'g--', label='recall')
    l3 = plt.plot(f1, 'b--', label='f1')
    plt.plot(acc, 'ro-', recall, 'g+-', f1, 'b^-')
    plt.show()
    """
    logger.info("show all metrics(micro)")
    logger.info("acc %s", acc)
    logger.info("recall %s", recall)
    logger.info("f1 %s", f1)
    
    
def test(logger, config, data_gen, dataloader, name, checkpoint):
    device, model = get_torch_model(
        BertLinerSoftmax, 
        model_config = {"model_name": config["model_name"], "loss_func": "ce", "label_num": config["label_num"]},
        load_checkpoint_path = checkpoint,
    )

    trainer = Worker(
        device = device,
        model = model, 
    )

    outputs, _ = trainer.rollout(dataloader)
    entity_outputs, entity_labels, offset_outputs = data_gen.decode(
        outputs, 
        data_gen.get_tokenize_length(name), 
        data_gen.get_raw_data_y(name),
    )
    metric = NERMetric(data_gen.get_raw_data_x(name), entity_labels, entity_outputs)
    logger.info("test metric is %s", metric.get_score())
    logger.info("test mean metric is %s", metric.get_mean_score())


def run_ner(config):
    setup_seed(42)
    if not os.path.exists(config["folder_path"]):
        os.makedirs(config["folder_path"])
    logger = init_logger(log_path = os.path.join(config["folder_path"], "output.log"))
    logger.info("global config %s", config)

    # data
    logger.info("prepare data")
    n_gpus = max(torch.cuda.device_count(), 1)
    data_gen = config["data_cls"](model_name = config["model_name"])
    data_gen.init_data(data_path = config["data_folder_name"])
    dataloader = data_gen.get_dataloader(batch_size = config["batch_size_per_gpu"] * n_gpus)
    logger.info("dataloader down")

    # train
    if config["if_train"]:
        logger.info("train start")
        train(logger, config, data_gen, dataloader["train"])
        logger.info("train end")

    # dev
    if config["if_select"]:
        logger.info("select start")
        select(logger, config, data_gen, dataloader["dev"], "dev")
        logger.info("select end")
    
    # test
    if config["if_test"]:
        logger.info("test start")
        test(logger, config, data_gen, dataloader["test"], "test", os.path.join(config["folder_path"], "model/best_model.pth"))
        logger.info("test end")
from dataloader.preprocessor.byte_ner import BYTETESTPreProcessor, BYTETrainPreProcessor
from worker.worker import Worker
from utils.torch_related import setup_seed


class BYTENERServer:
    def __init__(
        self,
        model_name,
        load_checkpoint_path,
    ):
        # kwargs
        self.model_name = model_name
        self.load_checkpoint_path = load_checkpoint_path

        # prepare model
        setup_seed(42)
        self.trainer = Worker(load_checkpoint_path=load_checkpoint_path)

    def predict(self, data):
        data_gen = BYTETESTPreProcessor(data_path=data, model_name=self.model_name)
        _, outputs = self.trainer.rollout(data_gen.get_dataloader["test"])
        entity_outputs, _ = data_gen.decode(
            outputs, 
            data_gen.get_tokenize_length("test"), 
        )
        return dict(zip(data_gen.get_raw_data_id, entity_outputs))

    def train(self, data):
        data_gen = BYTETrainPreProcessor(
            data_path=data, 
            model_name=self.model_name,
        )
        new_dataloader = data_gen.get_dataloader()
        self.trainer.save_checkpoint_path = self.load_checkpoint_path
        self.trainer.train_dataloader = new_dataloader["train"]
        self.trainer.dev_dataloader = new_dataloader["dev"]
        # 训练并更新模型
        self.trainer.train()
        del self.trainer
        self.trainer = Worker(load_checkpoint_path=self.load_checkpoint_path)
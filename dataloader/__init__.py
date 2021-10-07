from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from dataloader.ace05 import ACE05
from dataloader.ner_processor import NERProcessor, NERDataSet
from utils.data import dict_to_list_by_max_len


ACE05_config = {
    "task": "NER",
    "data_path": "/Users/bytedance/Desktop/nlp-dataset/ace05/json",
    "train_fn": "train.json",
    "test_fn": "test.json",
    "dev_fn": "dev.json",
    "ner_tag_method": "BIO",
    "cased": True,
}


NER_processor_config = {
    "tokenizer_model": "bert-base-cased",
    "tokenizer_cls": BertTokenizerFast,
    "is_split_into_words": True,
    "return_offsets_mapping": True,
    "padding": "max_length",
    "truncation": True,
    "max_length": None,
    "return_tensors": "pt",
    "language": "en",
}


def get_ner_dataloader(
    data_cls = ACE05,
    data_config = {},
    processor_cls = NERProcessor,
    processor_config = {},
    dataset_cls = NERDataSet,
    batch_size = 24,
    train_shuffle = True,
    num_workers = 0,
    collate_fn = dict_to_list_by_max_len,
):
    default_data_config = eval(data_cls.__name__ + "_config").copy()
    default_data_config.update(data_config)
    default_processor_config = NER_processor_config.copy()
    default_processor_config.update(processor_config)
    data = processor_cls(data_cls, default_data_config, **default_processor_config).get_tensor()
    # build dataset
    ner_dataset = {}
    for item in data:
        ner_dataset[item] = dataset_cls(**data[item])
    # build dataloader
    ner_dataloader = {}
    for item in ner_dataset:
        if item == "train":
            shuffle = train_shuffle
        else:
            shuffle = False
        ner_dataloader[item] = DataLoader(
            ner_dataset[item], 
            batch_size = batch_size, 
            shuffle = shuffle, 
            num_workers = num_workers,
            collate_fn = collate_fn,
        )
    return ner_dataloader
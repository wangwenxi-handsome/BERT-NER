from torch.utils.data import DataLoader
from dataloader.processor.ner_processor import NERProcessor, NERDataSet


def get_ner_dataloader(
    data_cls,
    data_config = {},
    processor_cls = NERProcessor,
    processor_config = {},
    dataset_cls = NERDataSet,
    batch_size = 24,
    train_shuffle = True,
    num_workers = 0,
    collate_fn = None,
    raw_data = True,
):
    processor = processor_cls(data_cls, data_config, **processor_config)
    data = processor.get_tensor()
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
    if raw_data:
        return ner_dataloader, {"data_list": processor.get_data(), "data_tensor": data, "ner_tag": processor.get_ner_tag()}
    else:
        return ner_dataloader, {}
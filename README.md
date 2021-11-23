# BERT-NER
a NER framework by BERT encoder which supports all language.

# Process
## Data
    labeling method is sequence tag such as BIO or BIEO.
### first step
    - raw data such as Dataset(cner, ace05) and other types of files in business scenarios.
    - related code: dataloader/preprocessor/base/RDataset
    - in: csv, txt, etc.
    - out: [{"x": sentence1, "y": label1, "id": 0}, {"x": sentence2, "y": label2, "id": 0}...]
    - sentence is a string such as "a Taidi dog is a cat". label is a int list such as [0, 1, 2, 0, 0, 1]. 0 means O, 1 means B-Animal, 2 means I-Animal.
### second step
    - tokenize, split data to train, dev, test and decode(convert model output to NER results).
    - related code: dataloader/tokenize/NERTokenize
    - in: [{"x": sentence1, "y": label1, "id": 0}, {"x": sentence2, "y": label2, "id": 0}...]
    - out: {train: data, dev: data, test: data}, each data format(BERT input) is {"input_ids": tensor, "position_embeddings": tensor, "mask_embeddings": tensor, label: list, "length": list}
### get dataloader
    - code: dataloader/preprocessor/base/BasePreProcessor.get_dataloader
    - BasePreProcessor wrap the RDataset and NERTokenize in the following two steps.
### custom new data
    - the repo only support ace05, cner and bytener three datasets.
    - inherit RDataset and rewrite _preprocess_data and classes method, for example dataloader/preprocessor/base/cner/CNERRDataset
    - inherit BasePreProcessor and rewrite _read_file and init_data method, for example dataloader/preprocessor/base/cner/CNERRDataset/CNERPreProcessor.
    - NERTokenize don't need to modify
## model
    - based on Transformer
    - code: model/BertLinerSoftmax/BertLinerSoftmax
    - BERT + Liner + Softmax + CrossEntryLoss
## worker(train and rollout)
    - code: worker/worker/Worker
## Main
    - code: scripts/run_ner.py
    - only modifying the kwargs can support different datasets and models.
    
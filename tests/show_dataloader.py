import sys
import os
sys.path.append(os.getcwd())
from dataloader import get_ner_dataloader
from model.EnBertLinerSoftmax import EnBertLinerSoftmax

model = EnBertLinerSoftmax()
my_dataloader = get_ner_dataloader()

for data in my_dataloader["train"]:
    output = model(**data)
    print(output)


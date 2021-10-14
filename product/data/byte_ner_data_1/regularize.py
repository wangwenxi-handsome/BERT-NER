import sys
import os
sys.path.append(os.getcwd())
import numpy as np

fold_path = "./product/data/byte_ner_data_1"
# 数据是闭区间
a = np.load(os.path.join(fold_path, "con_qualified.npy")).tolist()
b = np.load(os.path.join(fold_path, "con_unqualified.npy")).tolist()
c = np.load(os.path.join(fold_path, "con_unchecked.npy")).tolist()
print(len(a), len(b), len(c))
data = []
data.extend(a)
data.extend(b)
data.extend(c)
print(len(data))
np.save(os.path.join(fold_path, "raw_data.npy"), data)

data = np.load(os.path.join(fold_path, "raw_data.npy")).tolist()
print(type(data))
print(len(data))




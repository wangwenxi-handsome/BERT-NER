import sys
import os
sys.path.append(os.getcwd())
import numpy as np

fold_path = "./product/data/byte_ner1"
"""# 数据是闭区间
a = np.load(os.path.join(fold_path, "con_qualified1015.npy")).tolist()
b = np.load(os.path.join(fold_path, "con_unchecked1015.npy")).tolist()
c = np.load(os.path.join(fold_path, "con_unmarked1015.npy")).tolist()
print(len(a), len(b), len(c))

# data labeled
data_labeled = []
data_labeled.extend(a)
data_labeled.extend(b)
print(len(data_labeled))
np.save(os.path.join(fold_path, "raw_data.npy"), data_labeled)

# data unlabeled
np.save(os.path.join(fold_path, "unlabeled_data.npy"), c) """

""" data = np.load(os.path.join(fold_path, "raw_data.npy")).tolist()
print(data[0])

# to dict
data = np.load(os.path.join(fold_path, "unlabeled_data.npy"), allow_pickle=True).tolist()
print(data) """

a = [{"data": 1, "result":2}, {"data": 3, "result":2}]
np.save("test.npy", a)

b = np.load("test.npy", allow_pickle = True).tolist()
print(b)


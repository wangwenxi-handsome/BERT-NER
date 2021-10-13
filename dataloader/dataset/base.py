import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

from dataloader.processor.ner_processor import NERTAG


class BaseDataset:
    def __init__(
        self, 
        task,
        folder_path,
        train_fn = None,
        dev_fn = None,
        test_fn = None,
        cased = True,
        if_tag_first = True,
        test_rate = 0.1,
        dev_rate = 0.1,
        cross_valid = False,
        random_state = 42,
    ):
        self.task = task

        # common precess
        self.cased = cased
        self.if_tag_first = if_tag_first

        # for train split(if no dev)
        self.test_rate = test_rate
        self.dev_rate = dev_rate
        self.cross_vaild = cross_valid
        self.random_state = random_state

        # data path
        self.data_path = {}
        self.folder_path = folder_path
        self.data_path["train"] = os.path.join(folder_path, train_fn) if train_fn is not None else None
        self.data_path["test"] = os.path.join(folder_path, test_fn) if test_fn is not None else None
        self.data_path["dev"] = os.path.join(folder_path, dev_fn) if dev_fn is not None else None
        assert not (train_fn is None and test_fn is None and dev_fn is None)

    def data_precessor(self, folder_path):
        raise NotImplementedError(f"please implement the data_precessor func")

    def get_data(self):
        save_file_path = os.path.join(self.folder_path, "data.npy")
        if os.path.exists(save_file_path):
            data = np.load(save_file_path, allow_pickle=True).item()
        else:
            data = {}
            for i in self.data_path:
                if self.data_path[i] is not None:
                    data_x, data_y = self.data_precessor(self.data_path[i])
                    data[i] = {"x": data_x, "y": data_y}
            data = self.split_data(data)
            np.save(save_file_path, data)
        return data

    def split_data(self, data):
        # no train
        if "train" not in data:
            return data
        # have train
        else:
            # have dev and test
            if "dev" in data and "test" in data:
                return data
            else:
                # train_rate is used to keep sample rate
                train_rate = 1
                if "test" not in data and self.test_rate > 0:
                    data["test"] = {}
                    data["train"]["x"], data["test"]["x"], data["train"]["y"], data["test"]["y"] = train_test_split(
                        data["train"]["x"], 
                        data["train"]["y"], 
                        test_size = self.test_rate, 
                        random_state = self.random_state,
                    )
                    train_rate -= self.test_rate
                if "dev" not in data:
                    data["dev"] = {}
                    data["train"]["x"], data["dev"]["x"], data["train"]["y"], data["dev"]["y"] = train_test_split(
                        data["train"]["x"], 
                        data["train"]["y"], 
                        test_size = self.dev_rate / train_rate, 
                        random_state = self.random_state,
                    )
                return data

    def json_to_list(self, folder_path):
        data = []
        with open(folder_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data.append(json.loads(line))
        return data
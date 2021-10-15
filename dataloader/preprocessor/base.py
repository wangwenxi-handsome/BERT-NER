from sklearn.model_selection import train_test_split


class RDataset:
    def __init__(
        self, 
        split_rate = [],
        if_cross_valid = False,
        ner_tag_method = "BIO",
        cased = True,
        if_tag_first = True,
    ):
        # ner precess
        self.cased = cased
        self.ner_tag_method = ner_tag_method
        self.if_tag_first = if_tag_first

        # for train split(if no dev)
        self.split_rate = split_rate
        self.cross_vaild = if_cross_valid
    
    def get_ner_tag(self):
        return self.ner_tag

    def get_data_with_list_format(self, data):
        """preprocess dataset
        in: [{}, {}, {}...]
        out(get_data): [{"x": , "y": , "id": }, ...]
        """
        return self.split_data(self.preprocess_data(data))

    def split_data(self, data):
        """        
        in: {"x": , "y": , "id": }
        out: [{"x": , "y": , "id": }, ...]
        """
        if len(self.split_rate) == 0:
            splited_data = [data]
        else:
            # first split
            raw_rate = 1
            data1 = {}
            data2 = {}
            data1["x"], data2["x"], data1["y"], data2["y"], data1["id"], data2["id"] = train_test_split(
                data["x"], 
                data["y"],
                data["id"],
                test_size = self.split_rate[0], 
                random_state = 42,
            )
            raw_rate -= self.split_rate[0]

            if len(self.split_rate) == 1:
                splited_data = [data1, data2]
            elif len(self.split_rate) == 2:
                # second split
                data3 = {}
                data1["x"], data3["x"], data1["y"], data3["y"], data1["id"], data3["id"] = train_test_split(
                    data1["x"], 
                    data1["y"],
                    data1["id"],
                    test_size = self.split_rate[1] / raw_rate, 
                    random_state = 42,
                )
                splited_data = [data1, data2, data3]
            else:
                raise ValueError(f"len(split_rate) must <= 2")
        return splited_data

    def preprocess_data(self, data):
        """
        in: [{}, {}, {}...]
        out: {"x": , "y": , "id": } (not split)
        """
        raise NotImplementedError(f"please implement the data_precessor func")
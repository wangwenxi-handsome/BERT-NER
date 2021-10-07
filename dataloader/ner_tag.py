class NERTAG:
    def __init__(
        self,
        ner_class,
        ner_tag_method,
    ):
        self.ner_class = ner_class
        self.ner_tag_method = ner_tag_method

        self.tag2id = {"O": 0}
        i = 0
        for c in self.ner_class:
            for s in self.ner_tag_method:
                i += 1
                self.tag2id[c + "-" + s] = i
        self.id2tag = dict(zip(self.tag2id.values(), self.tag2id.keys()))

    def map_B2I(self, id):
        if id == 0:
            return 0
        tag = self.id2tag[id]
        if tag[-1] == "B":
            tag = tag[:-1] + "I"
        return self.tag2id[tag]
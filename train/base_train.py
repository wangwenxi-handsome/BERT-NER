class Trainer:
    def __init__(self, opt, model, train_data, dev_data, test_data, epochs=10):
        self.opt = opt
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            for data in self.train_data:
                data_x, data_y = data
                self.opt.zero_grad()
                output, loss = self.model(data_x, data_y)
                loss.backward()
                self.opt.step()
                print(loss)
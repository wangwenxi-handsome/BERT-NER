import os
import sys
sys.path.append(os.getcwd())
import numpy as np

from worker.server import BYTENERServer


if __name__ == "__main__":
    # init server
    server =BYTENERServer(
        model_name="bert-base-chinese", 
        load_checkpoint_path="product/data/byte_ner1/checkpoint/12.pth"
    )

    # test
    predict_data = np.load("product/data/byte_ner1/unlabeled_data.npy")
    results = server.predict(predict_data)
    print(results)

    # finetune
    train_data = np.load("product/data/byte_ner1/raw_data.npy")
    server.train(predict_data)

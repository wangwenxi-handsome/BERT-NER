import os
import sys
sys.path.append(os.getcwd())
import time
import numpy as np

from worker.server import BYTENERServer


if __name__ == "__main__":
    # init server
    server = BYTENERServer(
        model_name = "bert-base-chinese", 
        load_checkpoint_path = "/opt/tiger/nlp-xixi/product/experiments/byte1/model/best_model.pth",
        label_num = 63,
    )

    # test
    predict_data = [
        {"sentence": "北京市人民政府，天安门，故宫，长城，颐和园哪个最牛", "id": 0},
    ]
    t1 = time.time()
    results = server.predict(predict_data)
    print("predict时间", time.time() - t1)
    print(results)
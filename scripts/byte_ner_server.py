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
        load_checkpoint_path = None,
    )

    # test
    predict_data = [
        {"sentence": "北京市人民政府，天安门，故宫，长城，颐和园哪个最牛", "id": 0},
        {"sentence": "出行坐飞机，高铁，列车，大巴，公交，汽车，自行车哪个最快", "id": 1},
    ]
    t1 = time.time()
    # [{"itemID": [(class, start, end), (class, start, end)]}...]
    results = server.predict(predict_data)
    print("predict时间", time.time() - t1)
    print(results)
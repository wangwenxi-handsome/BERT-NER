import os
import sys
sys.path.append(os.getcwd())
import json
from collections import Counter
import numpy as np
import jieba


data_path = "/Users/bytedance/Desktop/nlp-dataset/byte/con_unchecked1021.npy"
train_path = "/Users/bytedance/Desktop/nlp-dataset/byte/sentence_train.txt"
test_path = "/Users/bytedance/Desktop/nlp-dataset/byte/sentence_test.txt"


def read_txt(dp):
    with open(dp, "r") as f:
        sentence = []
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            sentence.append(line)
    return sentence


def get_data(data, select_data):
    print("data to select", len(select_data))
    selected_data = []

    sentence = [i["sentence"].strip('\n') for i in data]
    sen_id = dict(zip(sentence, range(len(sentence))))
    for i, j in enumerate(select_data):
        if j in sen_id:
            tmp_data = data.copy()
            tmp_data[sen_id[j]]["id"] = i
            selected_data.append(tmp_data[sen_id[j]])

    print("selected data", len(selected_data))
    return selected_data


def get_user_dict():
    user_dict = {}

    data = np.load(data_path, allow_pickle = True).tolist()
    train_data = get_data(data, read_txt(train_path))
    test_data = get_data(data, read_txt(test_path))

    # get dict
    record_pos = {}
    for i, td in enumerate(train_data):
        for r in td["results"]:
            """
            if r[3] in user_dict and r[2] != user_dict[r[3]]:
                print(f"{r[3]} complict")
                print("sentence1")
                print(train_data[record_pos[r[3]]])
                print("sentence2")
                print(td)
            """
            user_dict[r[3]] = r[2]
            record_pos[r[3]] = i

    print("dict length", len(user_dict))
    return user_dict, train_data, test_data


def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def get_mean_score(class_info):
    score_dict = {"micro": class_info["all"]}
    recall_sum = 0
    precision_sum = 0
    f1_sum = 0
    for i in class_info:
        if i != "all":
            recall_sum += class_info[i]["recall"]
            precision_sum += class_info[i]["acc"]
            f1_sum += class_info[i]["f1"]
    score_dict["macro"] = {
        "acc": precision_sum / (len(class_info) - 1),
        "recall": recall_sum / (len(class_info) - 1),
        "f1": f1_sum / (len(class_info) - 1),
    }
    return score_dict


def get_result_from_txt(data_path):
    sentence = read_txt(data_path)
    results = []
    for d in sentence:
        words = jieba.lcut(d)
        result = []
        start = 0
        for w in words:
            if w in user_dict and len(w) > 2 and (w not in ['产品H食品','事件H项目策划','产品H设备工具','地点H公共场所','事件H节假日','产品H交通工具','软件系统H应用软件','组织H科研院校','软件系统H网站','组织H企业机构','地点H楼宇建筑物','产品H服饰','职业岗位','产品H文娱产品','地点Hother','组织H行政机构','产品H金融产品','组织Hother','技术术语H技术概念','规定H法律法规','技术术语H技术指标']):
                result.append((user_dict[w], start, start + len(w) - 1))
            start += len(w)
        results.append(result)
    np.save("/Users/bytedance/Downloads/unmark.npy", results, allow_pickle = True)

    
if __name__ == "__main__":
    user_dict, train_data, test_data = get_user_dict()
    jieba.load_userdict(list(user_dict.keys()))
    np.save("/Users/bytedance/Desktop/nlp-dataset/byte/test.npy", test_data, allow_pickle = True)

    # get unmark result from json
    get_result_from_txt('/Users/bytedance/Downloads/sentence_unmark.txt')

    # test
    labels = []
    for td in test_data:
        label = []
        for r in td["results"]:
            label.append((r[2], r[3]))
        labels.append(label)

    results = []
    results_tuple = []
    long_results = []
    for td in test_data:
        words = jieba.lcut(td["sentence"])
        labels_class = [i[3] for i in td["results"]]
        long_result = []
        result = []
        result_tuple = []
        start = 0
        for w in words:
            if w in user_dict:
                long_result.append((user_dict[w], w))
            if w in user_dict and w in labels_class:
                result.append((user_dict[w], w))
                result_tuple.append((user_dict[w], start, start + len(w) - 1))
            start += len(w)
        results.append(result)
        results_tuple.append(result_tuple)
        long_results.append(long_result)  

    # save
    outputs = []
    for i in range(len(results_tuple)):
        tmp_dict = {}
        tmp_dict["result"] = results_tuple[i]
        tmp_dict["id"] = test_data[i]["id"]
        outputs.append(tmp_dict)
    np.save("/Users/bytedance/Desktop/jieba_result.npy", outputs)
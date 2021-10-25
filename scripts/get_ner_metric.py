import numpy as np
from collections import Counter


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


if __name__ == "__main__":
    data = np.load("/Users/bytedance/Desktop/nlp-dataset/byte/test.npy", allow_pickle = True).tolist()
    tmp_results = np.load("/Users/bytedance/Downloads/combine/combine_result.npy", allow_pickle = True).tolist()
    results = []
    labels = []
    for i in range(len(data)):
        result = []
        label = []
        for j in data[i]["results"]:
            label.append((j[2], j[0], j[1] - 1))
        for j in tmp_results[i]['entities']:
            result.append((j[2], j[0], j[1]))
        results.append(result)
        labels.append(label)
    
    # metrics
    tmp_rights = []
    for i in range(len(results)):
        right = []
        for j in results[i]:
            if j in labels[i]:
                right.append(j)
        tmp_rights.append(right)

    # extend
    origins = []
    founds = []
    rights = []
    for s in range(len(labels)):
        origins.extend(labels[s])
        founds.extend(results[s])
        rights.extend(tmp_rights[s])

    # 找出每类的个数
    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        class_origin = count
        class_found = found_counter.get(type_, 0)
        class_right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(class_origin, class_found, class_right)
        if precision > 0:
            class_info[type_] = {"acc": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = compute(origin, found, right)
    class_info["all"] = {"acc": precision, "recall": recall, "f1": f1}
    mean_info = get_mean_score(class_info)
    for i in class_info:
        if i != "all":
            print(i, class_info[i])
    print(mean_info)
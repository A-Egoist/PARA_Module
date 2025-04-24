import numpy as np


def get_Recall(t, r):
    return len(set(t) & set(r)) / len(t)


def get_Precision(t, r):
    return len(set(t) & set(r)) / len(r)


def get_HR(t, r):
    return 1 if len(set(t) & set(r)) > 0 else 0


def get_RR(t, r):
    for index, item in enumerate(r):
        if item in t:
            return 1 / (index + 1)
    return 0


def get_AP(t, r):
    hits, sum_precision = 0, 0
    for index, item in enumerate(r):
        if item in t:
            hits += 1
            sum_precision += hits / (index + 1)
    if hits > 0:
        return sum_precision / hits
    else:
        return 0


def get_NDCG(t, r, top_k):
    idcg = 0.0
    idcg_length = min(len(t), top_k)
    for i in range(idcg_length):
        idcg += 1.0 / np.log2(i + 2)  # i start from 0, so need add 2 instead.
    dcg = 0.0
    for index, item in enumerate(r):
        if item in t:
            dcg += 1.0 / np.log2(index + 2)
    return dcg / idcg


def get_Novelty(r, item_popularity, top_k):
    sum_log = 0
    for i in r:
        sum_log += -np.log2(max(item_popularity[i], 1e-6))  # avoid log(0)
    return sum_log / top_k


def get_ARP(l, item_popularity):
    l_p = [item_popularity[i] for i in l]
    return np.mean(l_p)


def get_ARQ(l, item_quality):
    l_q = [item_quality[i] for i in l]
    return np.mean(l_q)
import numpy as np


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    print("relevance:")
    print(relevance)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    print("it2rel:")
    print(it2rel)
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    print("rank_scores:")
    print(rank_scores)
    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


l1 = [1,4,5]
l2 = [1,2,3]
l3 = [7,6,9]
l4 = [7,11,12]
l5 = [6, 3, 4]
l6 = [1, 2, 4]
a = getNDCG(l1, l2)
b = getNDCG(l3,l4)
c = getNDCG(l5,l6)
print(a)
print(b)
print(c)
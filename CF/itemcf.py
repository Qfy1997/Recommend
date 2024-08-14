#-*- coding: utf-8 -*-
'''
copy from Lockvictor,并在此基础上增加了两个数据集和一个NDCG评价标准。
'''
import sys
import random
import math
import os
from operator import itemgetter
import numpy as np
from collections import defaultdict

random.seed(0)


class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_item = 20
        self.n_rec_item = 20

        self.item_sim_mat = {}
        self.item_popular = {}
        self.item_count = 0

        print('Similar item number = %d' % self.n_sim_item, file=sys.stderr)
        print('Recommended item number = %d' %
              self.n_rec_item, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)
    
    def generate_dataset_for_yelp(self,filename,pivot=0.7):
        trainset_len = 0
        testset_len = 0
        i=0
        for line in self.loadfile(filename):
            if i == 0 :
                i+=1
                continue
            user, item = line.split('\t')
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(1)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(1)
                testset_len += 1
        print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)
    
    def generate_dataset_for_beauty(self,filename,pivot=0.7):
        trainset_len = 0
        testset_len = 0
        i=0
        for line in self.loadfile(filename):
            if i == 0 :
                i+=1
                continue
            user, item ,_= line.split('\t')
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(1)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(1)
                testset_len += 1
        print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)

    def calc_movie_sim(self):
        ''' calculate movie similarity matrix '''
        print('counting movies number and popularity...', file=sys.stderr)

        for user, items in self.trainset.items():
            for item in items:
                # count item popularity
                if item not in self.item_popular:
                    self.item_popular[item] = 0
                self.item_popular[item] += 1

        print('count items number and popularity succ', file=sys.stderr)

        # save the total number of movies
        self.item_count = len(self.item_popular)
        print('total item number = %d' % self.item_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.item_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        for user, items in self.trainset.items():
            for m1 in items:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in items:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print('calculating item similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_items in itemsim_mat.items():
            for m2, count in related_items.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.item_popular[m1] * self.item_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating item similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)

        print('calculate item similarity matrix(similarity factor) succ',file=sys.stderr)
        print('Total similarity factor number = %d' %simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar items and recommend N items. '''
        K = self.n_sim_item
        N = self.n_rec_item
        rank = {}
        recommended_items = self.trainset[user]
        for movie, _ in recommended_items.items():
            for related_item, similarity_factor in sorted(self.item_sim_mat[movie].items(),key=itemgetter(1), reverse=True)[:K]:
                if related_item in recommended_items:
                    continue
                rank.setdefault(related_item, 0)
                rank[related_item] += similarity_factor    # * rating
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_item
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_items = self.testset.get(user, {})
            rec_items = self.recommend(user)
            for item, _ in rec_items:
                if item in test_items:
                    hit += 1
                all_rec_items.add(item)
                popular_sum += math.log(1 + self.item_popular[item])
            rec_count += N
            test_count += len(test_items)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.item_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %(precision, recall, coverage, popularity), file=sys.stderr)

def getDCG(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    idcg = getDCG(relevance)
    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


if __name__ == '__main__':
    # ratingfile = os.path.join('../dataset/ml-1m', 'ratings.dat')
    yelpfile = os.path.join('../dataset/yelp','Yelp.inter')
    # beautyfile = os.path.join('../dataset/Beauty','Beauty.inter')
    itemcf = ItemBasedCF()
    # itemcf.generate_dataset(ratingfile)
    # itemcf.generate_dataset_for_beauty(beautyfile)
    itemcf.generate_dataset_for_yelp(yelpfile)
    itemcf.calc_movie_sim()
    itemcf.evaluate()
    # NDCG评价指标
    print("NDCG start....")
    NDCG_lis=[]
    for i, user in enumerate(itemcf.trainset):
        # print("processing...:",i)
        test_movies = itemcf.testset.get(user, {})
        rec_movies = itemcf.recommend(user)
        pos_lis=[]
        rec_lis=[]
        for item in test_movies:
            pos_lis.append(int(item))
        for item in rec_movies:
            rec_lis.append(int(item[0]))
        # print(pos_lis)
        # print(rec_lis)
        ncdg=getNDCG(rec_lis,pos_lis)
        NDCG_lis.append(ncdg)
    NDCG_len=len(NDCG_lis)
    print(len(NDCG_lis))
    NDCG=0
    for item in NDCG_lis:
        NDCG+=item
    NDCG=NDCG/NDCG_len
    print("NDCG=",NDCG)

    # movielens
    # n_sim_item=10  n_rec_item=10
    # precision@10=0.3844  recall@10=0.0774  coverage=0.1983 popularity=7.0540  NDCG@10=0.1790
    # n_sim_item=20 n_rec_item=10
    # precision=0.3792 recall@10=0.0764   coverage=0.1729 popularity=7.1730
    # n_sim_item=20  n_rec_item=20
    # precision@20=0.3239  recall@20=0.1305  coverage=0.2433  popularity=7.0709  NDCG@20=0.2318

    # yelp
    # n_sim_item=10  n_rec_item=10
    # precision@10=0.0438  recall@10=0.0297  coverage=0.6199 popularity=3.6318  NDCG@10=0.0396
    # n_sim_item=20 n_rec_item=10
    # precision=0.0493  recall@10=0.0334  coverage=0.5862  popularity=3.8245 NDCG@10=
    # n_sim_item=20  n_rec_item=20
    # precision@20=0.0399  recall@20=0.0540  coverage=0.7501  popularity=3.7274  NDCG@20=

    # Beauty
    # n_sim_item=10  n_rec_item=10
    # precision@10=0.0175  recall@10=0.0657  coverage=0.9006 popularity=2.0509  NDCG@10=0.0421
    # n_sim_item=20 n_rec_item=10
    # precision=0.0177 recall@10=0.0663   coverage=0.9085 popularity=2.0832  NDCG@10=0.0428
    # n_sim_item=20  n_rec_item=20
    # precision@20=0.0127  recall@20=0.0951  coverage=0.9792  popularity=2.0959  NDCG@20=0.0499

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


class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering ''' 
    def __init__(self):
        self.trainset = {}
        self.testset = {}
        self.n_sim_user = 20
        self.n_rec_item = 20
        # usersim_mat为一个字典，每个key对应一个用户及其对应的一个字典，该字典存放与该用户共同推荐过该item的用户以及对应次数。
        self.user_sim_mat = {}
         # 每个key对应一个item的id，value对应被推荐过的的次数
        self.item_popular = {}
        self.item_count = 0

        print('Similar user number = %d'% self.n_sim_user, file=sys.stderr)
        print('recommended item number = %d'%self.n_rec_item, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        for line in self.loadfile(filename):
            user, item, rating, _ = line.split('::')
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(rating)
                testset_len += 1

        print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)
    
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

    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=itemID, value=list of userIDs who have recommend this item
        print ('building movie-users inverse table...', file=sys.stderr)
        # 初始化一个item2users的一个dict，每个item的id作为key，对应value存放推荐过这个item对应的用户id
        item2users = dict()
        # 遍历训练集
        for user, items in self.trainset.items():
            # 遍历每个用户推荐过的item
            for item in items:
                # inverse table for item-users
                if item not in item2users:
                    item2users[item] = set()
                item2users[item].add(user)
                # count item popularity at the same time
                if item not in self.item_popular:
                    self.item_popular[item] = 0
                
                self.item_popular[item] += 1
        print('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        # 被用户推荐过的item
        self.item_count = len(item2users)
        print('total movie number = %d' % self.item_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print('building user co-rated movies matrix...', file=sys.stderr)
        # usersim_mat为一个字典，每个key对应一个用户及其对应的一个字典，该字典存放与该用户共同推荐过的item的用户以及对应次数。
        # 存放用户u和用户v共同推荐过同一个item的次数
        for item, users in item2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print('build user co-rated movies matrix succ', file=sys.stderr)
        # calculate similarity matrix
        print('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
        
        # 遍历每个用户及其共同推荐过item的用户
        for u, related_users in usersim_mat.items():
            # 遍历与用户u共同推荐过item的用户v及其次数count
            for v, count in related_users.items():
                # usersim_mat[u][v]的值等于 count 与 len(trainset[u])和len(trainset[v])的乘积的平方根 的 比值
                # len(trainset[u])表示:用户u推荐过item的总次数
                # len(trainset[v])表示:用户v推荐过item的总次数
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating user similarity factor(%d)'%simfactor_count, file=sys.stderr)
        print('calculate user similarity matrix(similarity factor) succ',file=sys.stderr)
        print('Total similarity factor number = %d'%simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user
        N = self.n_rec_item
        rank = dict()
        reced_items = self.trainset[user]
        # 遍历K个与当前user相似度最高的用户及其对应的分数
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),key=itemgetter(1), reverse=True)[0:K]:
            # 遍历当前similar_user推荐过的item
            for item in self.trainset[similar_user]:
                # 推荐过的item不做推荐，即不添加到rank中
                if item in reced_items:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(item, 0)
                # 对当前item的分数进行追加similarity_factor操作
                rank[item] += similarity_factor
        # return the N best items
        # 根据similarity_factor返回前N个item
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

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
            # pos_lis=[]
            # rec_lis=[]

            for item, _ in rec_items:
                if item in test_items:
                    hit += 1
                all_rec_items.add(item)
                popular_sum += math.log(1 + self.item_popular[item])
            
            rec_count += N
            test_count += len(test_items)
            

        # 推荐对的个数与所有推荐的东西的个数的比值
        precision = hit / (1.0 * rec_count)
        # 推荐对的个数与测试集中的个数的比值
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
    ratingfile = os.path.join('../dataset/ml-1m', 'ratings.dat')
    yelpfile=os.path.join('../dataset/yelp','Yelp.inter')
    beautyfile=os.path.join('../dataset/Beauty','Beauty.inter')
    # print(ratingfile)
    usercf = UserBasedCF()
    # 从ratings文件中生成数据集,取一个随机数，大于0.7放入测试集，小于0.7放入训练集
    # 训练集中key存放用户，value存放该用户对看过的电影及评价分数一个字典
    usercf.generate_dataset(ratingfile)
    # usercf.generate_dataset_for_yelp(yelpfile)
    # usercf.generate_dataset_for_beauty(beautyfile)
    usercf.calc_user_sim()
    usercf.evaluate()
    # NDCG评价指标
    NDCG_lis=[]
    for i, user in enumerate(usercf.trainset):
        test_movies = usercf.testset.get(user, {})
        rec_movies = usercf.recommend(user)
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

    # movielens:
    # n_sim_user = 10 n_rec_movie = 10
    # precision@10=0.3373  recall@10=0.0680 coverage=0.4094 popularity=6.7834 NDCG@10=0.16
    # n_sim_user = 20 n_rec_movie = 10
    # precision@10=0.3766  recall@10=0.0759  coverage=0.3175  popularity=6.9196 NDCG@10=0.18
    # n_sim_user = 20 n_rec_movie = 20
    # precision@20=0.3226  recall@20=0.1300  coverage=0.4048 popularity=6.8118 NDCG@20=0.24

    # yelp:
    # n_sim_user = 10 n_rec_movie = 10
    # precision@10=0.0492  recall@10=0.0333  coverage=0.6772 popularity=4.6624  NDCG@10=0.04
    # n_sim_user = 20 n_rec_movie = 10
    # precision@10=0.0580  recall@10=0.0392  coverage=0.4960 popularity=4.9356  NDCG@10=0.049
    # n_sim_user = 20 n_rec_movie = 20
    # precision@20=0.0472  recall@20=0.0639  coverage=0.6945 popularity=4.7900  NDCG@20=0.0645

    # Beauty:
    # n_sim_user = 10 n_rec_movie = 10
    # precision@10=0.0155  recall@10=0.0583  coverage=0.9416 popularity=3.0720  NDCG@10=0.0367
    # n_sim_user = 20 n_rec_movie = 10
    # precision@10=0.0182  recall@10=0.0682  coverage=0.9308 popularity=3.1598  NDCG@10=0.0451
    # n_sim_user = 20 n_rec_movie = 20
    # precision@20=0.0127  recall@20=0.0955  coverage=0.9802 popularity=3.0796  NDCG@20=0.0521    

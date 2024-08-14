import torch
import torch.nn as nn


"""
BPR Loss是用得比较多的一种raking loss。
它是基于Bayesian Personalized Ranking。
BPR Loss 的思想很简单，就是让正样本和负样本的得分之差尽可能达到最大
"""
class BPR_Loss(nn.Module):
    def __init__(self,
                 batch_size:int,
                 decay_ratio:float=1e-5):
        super(BPR_Loss, self).__init__()

        self.batch_size= batch_size
        self.decay = decay_ratio

    def forward(self,users,pos_items,neg_items):
        # users的embedding与pos_item的embedding进行点乘并求和得到最终的pos_scores
        pos_scores = torch.mul(users,pos_items).sum(dim=1)
        # users的embedding与neg_item的embedding进行点乘并求和得到最终的neg_scores
        neg_scores =  torch.mul(users,neg_items).sum(dim=1)

        log_prob = nn.LogSigmoid()(pos_scores - neg_scores).sum()

        regularization = self.decay*(users.norm(dim=1).pow(2).sum()
                                     +pos_items.norm(dim=1).pow(2).sum()
                                     +neg_items.norm(dim=1).pow(2).sum())
        loss =  regularization-log_prob
        loss /=self.batch_size
        return loss


    def __call__(self,*args):
        return self.forward(*args)



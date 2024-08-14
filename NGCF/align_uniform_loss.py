import torch
import torch.nn as nn
import torch.nn.functional as F


"""
align_uniform_loss
"""
class align_uniform_Loss(nn.Module):
    def __init__(self,
                 batch_size:int,
                 decay_ratio:float=1e-5):
        super(align_uniform_Loss, self).__init__()

        self.batch_size= batch_size
        self.decay = decay_ratio

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        #闵可夫斯基距离(Minkowski Distance) p=2时为欧氏距离
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self,users,pos_items,neg_items):
        user_e_n, item_e_n = F.normalize(users, dim=-1), F.normalize(pos_items, dim=-1)
        # users的embedding与pos_item的embedding进行点乘并求和得到最终的pos_scores
        # pos_scores = torch.mul(users,pos_items).sum(dim=1)
        # users的embedding与neg_item的embedding进行点乘并求和得到最终的neg_scores
        # neg_scores =  torch.mul(users,neg_items).sum(dim=1)

        # log_prob = nn.LogSigmoid()(pos_scores - neg_scores).sum()

        # regularization = self.decay*(users.norm(dim=1).pow(2).sum()+pos_items.norm(dim=1).pow(2).sum()+neg_items.norm(dim=1).pow(2).sum())

        align = self.alignment(user_e_n, item_e_n).sum()
        uniform = 0.5 * (self.uniformity(user_e_n) + self.uniformity(item_e_n)) / 2
        uniform = uniform.sum()
        # align /=self.batch_size
        # uniform /=self.batch_size
        # loss =  regularization-log_prob+align+uniform
        loss = align+uniform
        loss /=self.batch_size

        return loss


    def __call__(self,*args):
        return self.forward(*args)



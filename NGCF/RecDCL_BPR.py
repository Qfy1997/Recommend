import torch
import torch.nn as nn
import torch.nn.functional as F


"""
RecDCL_loss
"""
class RecDCL_BPR_loss(nn.Module):
    def __init__(self,
                 batch_size:int,
                 decay_ratio:float=1e-5):
        super(RecDCL_BPR_loss, self).__init__()

        self.batch_size= batch_size
        self.decay = decay_ratio
        self.predictor = nn.Linear(2048, 2048)
        self.bn = nn.BatchNorm1d(2048, affine=False)

        layers = []
        embs = str(2048) + '-' + str(2048) + '-' + str(2048)
        sizes = [2048] + list(map(int, embs.split('-')))
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)


    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def bt(self, x, y):
        bt_coeff = 0.01
        user_e = self.projector(x) 
        item_e = self.projector(y) 
        c = self.bn(user_e).T @ self.bn(item_e)
        c.div_(user_e.size()[0])
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(2048)
        # 取出c的非对角元素的值，取平方，求和，与embedding做比值。
        off_diag = self.off_diagonal(c).pow_(2).sum().div(2048)
        bt = on_diag + bt_coeff * off_diag # --bt_coeff 0.01
        return bt

    def poly_feature(self, x):
        polyc = 1e-7
        degree = 4
        a = 1
        user_e = self.projector(x) 
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (a * xx + polyc) ** degree # polyc 1e-7 self.degree = 4
        return poly.mean().log()

    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


    def forward(self,user_e, item_e, lightgcn_all_embeddings, u_target, i_target,neg_item_embedding):
        coeff = 5
        poly_coeff = 0.2
        user_e_n, item_e_n = F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

        user_e, item_e = self.predictor(user_e), self.predictor(item_e)
        bt_loss = self.bt(user_e_n, item_e_n)
        poly_loss = self.poly_feature(user_e_n) / 2 + self.poly_feature(item_e_n) / 2 
        mom_loss = self.loss_fn(user_e, i_target) / 2 + self.loss_fn(item_e, u_target) / 2

        # users的embedding与pos_item的embedding进行点乘并求和得到最终的pos_scores
        pos_scores = torch.mul(user_e,item_e).sum(dim=1)
        # users的embedding与neg_item的embedding进行点乘并求和得到最终的neg_scores
        neg_scores =  torch.mul(user_e,neg_item_embedding).sum(dim=1)

        log_prob = nn.LogSigmoid()(pos_scores - neg_scores).sum()

        regularization = self.decay*(user_e.norm(dim=1).pow(2).sum()
                                     +item_e.norm(dim=1).pow(2).sum()
                                     +neg_item_embedding.norm(dim=1).pow(2).sum())
        bpr_loss =  regularization-log_prob
        bpr_loss /=self.batch_size

        loss =  bpr_loss+bt_loss + poly_loss * poly_coeff + mom_loss * coeff # coeff=5 poly_coeff 0.2 

        return loss


    def __call__(self,*args):
        return self.forward(*args)



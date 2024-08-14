import torch
from datetime import datetime
from utils import LoadDataset
from utils import DatasetSplit
from laplacian_mat import Laplacian
from evaluation import Evaluation
from torch.utils.data import DataLoader
from model.NGCF import NGCF
from bpr_loss import BPR_Loss
# from bpr_ua_loss import BPR_ua_Loss
from RecDCL_loss import RecDCL_loss
# from RecDCL_BPR import RecDCL_BPR_loss
from align_uniform_loss import align_uniform_Loss
from train import Train
from parser import args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# 初始化数据集位置
root_path = '../dataset'
# 加载数据集
dataset = LoadDataset(root=root_path,file_category=args.file_category)
# 对数据集进行分割
total_df , train_df, test_df = dataset.split_train_test()
# 取得userId和itemId的索引
num_user, num_item = total_df['userId'].max()+1, total_df['itemId'].max()+1
# 训练集负采样
train_set = DatasetSplit(df=train_df,total_df=total_df,train=True,ng_ratio=4)
# 测试集负采样
test_set = DatasetSplit(df=test_df,total_df=total_df,train=False,ng_ratio=2)

# 生成拉普拉斯矩阵
matrix_generator = Laplacian(df=total_df)

# 对生成的拉普拉斯矩阵进行归一化
eye_matrix,norm_laplacian  = matrix_generator.create_norm_laplacian()

train_loader = DataLoader(train_set,
                          batch_size=args.batch,
                          shuffle=True)

test_loader = DataLoader(test_set,
                         batch_size=100,
                         shuffle=False,
                         drop_last=True
                         )

model = NGCF(norm_laplacian=norm_laplacian,
             eye_matrix= eye_matrix,
             num_user=num_user,
             num_item=num_item,
            #  embed_size=64,
            embed_size=2048,
             device= device,
             node_dropout_ratio=0.1,
             mess_dropout=[0.1,0.1,0.1],
             layer_size=3,
             )

model.to(device)
# criterion = BPR_Loss(batch_size=256,decay_ratio=1e-5)
# criterion = BPR_ua_Loss(batch_size=256,decay_ratio=1e-5)
criterion = RecDCL_loss(batch_size=256,decay_ratio=1e-5)
# criterion = RecDCL_BPR_loss(batch_size=256,decay_ratio=1e-5)
# criterion = align_uniform_Loss(batch_size=256,decay_ratio=1e-4)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)



if __name__ =='__main__' :
    start_time = datetime.now()
    print('------------train start------------')
    train = Train(device=device,
                  epochs=args.epoch,
                  model=model,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  optim=optimizer,
                  criterion=criterion,
                  top_k=args.top_k,
                  )
    train.train()
    print('------------train end------------')

    eval = Evaluation(test_dataloader=test_loader,
                      model=model,
                      top_k=args.top_k,
                      device=device,)
    HR,NDCG, Recall= eval.get_metric()
    print("HR@20:",HR,", NDCG@20:",NDCG,", Recall@20:",Recall)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

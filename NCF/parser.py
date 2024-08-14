import argparse
"""
这是一个初始参数解析文件
"""

parser=argparse.ArgumentParser(description="Run selected model")
parser.add_argument('-e','--epoch',type=int,default=30,help="Number of epochs") #规定训练的循环数
parser.add_argument('-b','--batch',type=int,default=256,help="Batch size") # 训练批次
parser.add_argument('-l','--layer', nargs='+',type=list,default=[64,32,16],help='MLP layer factor list') # MLP每层的参数
parser.add_argument('-f','--factor',type=int,default=8,help='choose number of predictive factor') # 每个itemId或userId对应的特征为factor维
parser.add_argument('-m','--model',type=str,default='NeuMF',help='select among the following model,[MLP, GMF, NeuMF]') # 选择需要训练的模型
parser.add_argument('-lr', '--lr', default=1e-3, type=float,help='learning rate for optimizer') # 学习率
parser.add_argument('-pr','--use_pretrain',type=str,default='False',help='use pretrained model or not') # 是否使用预训练完整的模型
parser.add_argument('-save','--save_model',type=str,default='True',help='save trained model or not') # 是否保存模型
parser.add_argument('-k','--topk',type=int,default=20,help='choose top@k for NDCG@k, HR@k, Recall@k') # 模型评价时需要取多少个样本
parser.add_argument('-fc','--file_category',type=str,default='1m',help='choose file category, [1m,Beauty,yelp]') # 选择需要进行训练及评价的数据集。
args = parser.parse_args()
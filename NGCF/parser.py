import argparse

parser= argparse.ArgumentParser(description="Run selected model")
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch',type=int,default=256,help="Batch size")
parser.add_argument('-lr', '--lr', default=1e-3, type=float,help='learning rate for optimizer')
parser.add_argument('-dl','--download',type=str,default='True',help='Download or not')
parser.add_argument('-k','--top_k',type=int,default=10,help='choose top@k for NDCG@k, HR@k')
parser.add_argument('-fc','--file_category',type=str,default='1m',help='choose file size, [1m,Beauty,yelp]')
args = parser.parse_args()
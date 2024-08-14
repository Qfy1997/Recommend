import numpy as np
import torch

# Hint Ratio评价标准
def hit(gt_item, pred_items):
	# 如果预测的样本在gt_item中
	if gt_item in pred_items:
		return 1
	return 0

# NDCG 评价标准
def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k, device):
	HR, NDCG, Recall = [], [],[]
	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)
		predictions = model(user, item)
		pre, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()
		gt_item = item[0].item()
		rec_labels = torch.take(label,indices).cpu().numpy().tolist()
		pre_list = pre.detach().numpy().tolist()
		new_pre_list=[]
		# 这里对pre_list做一个标签优化，模型通过sigmoid计算出的概率值大于0.9时，赋予label为1，否则为0
		for item in pre_list:
			if item>0.9:
				new_pre_list.append(1.0)
			else:
				new_pre_list.append(0.0)
		count = 0
		for i in range(len(new_pre_list)):
			if new_pre_list[i]==rec_labels[i]:
				count+=1
		# 计算每个批次的Recall值
		Recall.append(count/top_k)
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG),np.mean(Recall)

# Neural Graph  based model复现

在此作者的[代码](https://github.com/changhyeonnam/NGCF)基础之上，对lightGCN、DirectAU、RecDCL模型进行复现，并增加了BPR损失函数和Recall评价标准。

## 运行模型命令行

```java
python3 main.py -e 10 -b 256 -dl true -k 20 -fi 1m
```

## Reference

1. [Neural Graph  Collaborative Filtering](https://arxiv.org/abs/1905.08108)
2. [Official code from Xiang Wang](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)

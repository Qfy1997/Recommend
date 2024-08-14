## NCF (Neural Collaborative Filtering)模型复现

### 模型结果

可以产看log.txt文件来查看相关信息。

### 模型运行命令行

- 运行GMF模型

  ```python
  python main.py --epoch 30 --batch 256 --factor 8 --model GMF --topk 20 --file_size 1m --download True --save True

  ```
- 运行MLP模型

  ```python
  python main.py --epoch 30 --batch 256 --factor 8 --model MLP --topk 20 --file_size 1m --layer 64 32 16 --download False --save True

  ```
- 运行NeuMF模型

  ```python
  python main.py --epoch 30 --batch 256 --factor 8 --model NeuMF  --topk 20 --file_size 1m --layer 64 32 16 --download False --use_pretrain True
  ```

## Reference

1. [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
2. Official [code](https://github.com/hexiangnan/neural_collaborative_filtering) from author
3. 感谢此作者的[代码](https://github.com/changhyeonnam/NCF)贡献

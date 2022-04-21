# algorithmServer

用falsk来搭建算法api的server 现在有公式匹配和文本匹配环境如下：
```
jieba 0.42.1
flask 2.0.3
gensim 3.8.3
tensorflow-gpu 1.14.0
sympy 1.8
```
`function_matching`文件夹下是公式匹配算法
`function_matching`文件夹下是文本匹配算法
- `input`：训练数据和输入数据
- `model`：保存模型
    - `args.py`：保存超参数
    - `graph.py`：模型搭建
    - `predict.py`：模型预测
    - `train.py`：模型训练
    - `test.py`：模型测试
- `input`：训练数据和输入数据
- `output`：输出文件
- `utils`：保存工具函数

首先运行`word2vec_static.py`生成词向量，由于数据集不同所以需要训练不同的Word2Vec模型，然后运行`predict.py`进行模型的预测，运行`test.py`测试模型效果，训练时认为相似度大于0.5则匹配。
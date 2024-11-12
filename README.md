# BertPerplexity

用 Bert 计算句子的困惑度

主要原因是 ASR 总会不可避免的收到一些乱七八糟的识别结果，因此通过用 Bert 计算困惑度的方式来判断句子的流畅度，可以在语义层面加一个简单的拒识

(理论上只有自回归模型才有语言模型的概率定义，也才有困惑度的定义，所以这里的语言模型就不是常见的那种 n-gram 定义，而是 Bert 预训练的语言模型的定义)

如果只需要用困惑度来拒识效果是不错的，但有很多可以改进的地方，比如有一种情况，就是正确的指令前后跟了无意义的句子，例如 `三块今天天气怎么样马上送过来`, 这种情况可以做一个 LSTM 分类模型简单分一下

## 安装

```
pip install git+https://github.com/suzakuwcx/bertperplexity.git
```

## 用法

```python

import torch
from transformers import BertTokenizer

from bertperplexity import BertPerplexity

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = BertPerplexity.from_pretrained("google-bert/bert-base-chinese")
model = model.to(torch.device("cpu"))

with torch.no_grad():
    input_ids = tokenizer("今天天气怎么这么差")
    # 正常情况，tensor(1.3521)
    print(model(**input_ids))
    input_ids = tokenizer("明天什么天气没款的这些都可以")
    # ASR 乱识别范例，tensor(17.3301)
    print(model(**input_ids))

```

model(**input_ids) 返回的就是困惑度，模型可以去 hugginface 找一个你觉得不错的，修改 "google-bert/bert-base-chinese" 即可，只要是基于 Bert 的就行

关于拒识的阈值的话，可以先收集些业务数据的 ASR 识别结果，人工标记是否该拒识，然后通过修改阈值的方式去计算 AUC，选个 AUC 大的结果作为拒识阈值
AUC 计算参考 `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score`

# 原理

简单来说，和 Bert 预训练过程一致，Bert 建模的方式是通过随机在句子中替换 token 为 [MASK] 的方式，来预测 Mask 对应的 token, 所以我们可以通过对句子中间的部分替换为 [Mask], 让 Bert 计算出这个 token 出现的概率，通过这个概率来算困惑度

假如给一个句子 `今天天气怎么这么差` ， 首先掐去开头的 [CLS] 和末尾的 [SEP]， 然后掐去第一个字和最后一个字(因为 Bert 是双向注意力，边缘字的困惑度往往都比较大)，如下

```
今[MASK]天气怎么这么差
今天[MASK]气怎么这么差
...
今天天气怎么这[MASK]差
```

然后算出第一个句子中 [MASK] 预测为 `天` 的概率
...
最后一个句子中 [MASK] 预测为 `么` 的概率



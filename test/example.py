import torch
from transformers import BertTokenizer

from bertperplexity import BertPerplexity

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = BertPerplexity.from_pretrained("google-bert/bert-base-chinese")
model = model.to(torch.device("cpu"))

with torch.no_grad():
    input_ids = tokenizer("今天天气怎么这么差")
    print(model(**input_ids))

    input_ids = tokenizer("明天什么天气没款的这些都可以")
    print(model(**input_ids))

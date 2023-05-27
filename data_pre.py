import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import copy
import time
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
# from torchtext import data
# from torchtext.vocab import Vectors

train_df = pd.read_csv("./data/data_train.txt", sep="_!_", header=None, names=["text", "label"], encoding="utf-8")
test_df = pd.read_csv("./data/data_test.txt", sep="_!_", header=None, names=["text", "label"], encoding="utf-8")
val_df = pd.read_csv("./data/data_val.txt", sep = "_!_", header=None, names=["text", "label"], encoding="utf-8")
# print(train_df.head(5).text)

stop_words = pd.read_csv("./cn_stopwords.txt", header=None, names=["text"])
# print(stop_words)

voc_dict = {}
def chinese_pre(text_data):
    # 字符转化为小写，去除数字
    text_data = re.sub("\d+", "", str(text_data))
    text_data = text_data.lower()
    # 分词，使用精确模式
    text_data = list(jieba.cut(text_data, cut_all=False))
    # 去停用词和多余空格
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    for word in text_data:
        if word in voc_dict.keys():
            voc_dict[word] = voc_dict[word] + 1
        else:
            voc_dict[word] = 1
    text_data = " ".join(text_data)
    return text_data


# 对数据进行分词
train_df["cutword"] = train_df.text.apply(chinese_pre)
val_df["cutword"] = val_df.text.apply(chinese_pre)
test_df["cutword"] = test_df.text.apply(chinese_pre)

min_seq = 1
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                   key=lambda x:x[1], 
                   reverse=True)[:top_n]


voc_dict = {word_count[0]:idx for idx, word_count in enumerate(voc_list)}
voc_dict.update({UNK:len(voc_dict), PAD:len(voc_dict) + 1})
print(voc_dict)

# 预处理后的结果保存为新的文件
train_df[["label", "cutword"]].to_csv("./data/nd_train.csv", index=False)
test_df[["label", "cutword"]].to_csv("./data/nd_test.csv", index=False)
val_df[["label", "cutword"]].to_csv("./data/nd_val.csv", index=False)

train_df = pd.read_csv("./data/nd_train.csv")
test_df = pd.read_csv("./data/nd_test.csv")
val_df = pd.read_csv("./data/nd_val.csv")

ff = open("./dict", "w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))
ff.close()
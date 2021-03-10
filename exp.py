import json
import pickle
import time
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import nltk
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max

from data_utils import (make_graph_vector)
import config

path = os.path.abspath(".")
a = open(path + "\\data\\relation.txt", "r").readlines()
b = open("./data/relation.txt", "r").readlines()
print(a[1])
print(b[1])
e = []
c = [("0.05", "-1", "-0.2"), ("2", "3", "4")]
for d in c:
    e.append((float(d[1]), float(d[2]), float(d[0])))
print(e)

f = "hello, i, love, you"
j, k, l, m = f.split(", ")
print(j)
print(k)
print(l)
print(m)
g = f + "\t" + str(e)
print(g)

x = [1]
with open(config.relation_embedding, "rb") as f1:
    rel_embedding = pickle.load(f1)
    rel_embedding = torch.tensor(rel_embedding, dtype=torch.float)
relation_embedding = nn.Embedding(config.relation_size + 2, config.graph_vector_size). \
    from_pretrained(rel_embedding, freeze=True)

with open(config.entity_embedding, "rb") as f2:
    ent_embedding = pickle.load(f2)
    ent_embedding = torch.tensor(ent_embedding, dtype=torch.float)
entity_embedding = nn.Embedding(config.entity_size + 1, config.graph_vector_size). \
    from_pretrained(ent_embedding, freeze=True)

z = torch.LongTensor([[[1, 1], [2, 2], [3, 3]], [[2, 2], [3, 3], [4, 4]]])  # 2*3*2*1
a = torch.LongTensor([[[1, 1], [1, 1], [1, 1]], [[0, 0], [1, 1], [0, 0]]])  # 2*3*2
b = torch.BoolTensor([[[1, 1], [1, 1], [1, 1]], [[0, 0], [1, 1], [0, 0]]])
y = torch.LongTensor([21471])
p = torch.LongTensor(32, 251, 25, 3)
# print(torch.matmul(z.view(6,2,1),a.view(6,1,2)).view(2,3,2,2))
d = []
# with open("./data/paracs-dev.json", "r") as f:
# print(b)
a = a.masked_fill(b, value=2)
# print(a)
m = nn.Linear(100, 7, bias=False)
ou = m(relation_embedding(a) + entity_embedding(z)).view(12, 1, -1)
# print(ou)
ous = torch.tanh(ou)
ht = entity_embedding(p[:, :, :, 0])
print(ht)
b = b.bool()
z = z.masked_fill(b, value=0)
print(z)
dir = os.path.join("./result/cs_ans/", "generated.txt")
print(dir)

# print(relation_embedding(a[:, :, 1]).size())

# assert len(x) == len(y)
# x, y = [], []
# print(x, "\n", y)
# make_graph_vector(config.entity_embedding, config.relation_embedding)

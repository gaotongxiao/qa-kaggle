import torch
import numpy as np

idx2word = []
embeddingList = []
with open("data/glove.6B.100d.txt") as f:
    while True:
        line = f.readline()
        if not line: break
        line = line.split()
        idx2word.append(line[0])
        embeddingList.append(line[1:])
embedding = np.concatenate(np.array(embeddingList), axis=0).reshape(-1, 100)
print(embedding.shape)

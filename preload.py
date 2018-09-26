import pickle
import torch
import numpy as np

class words:
    def __init__(self):
        self.idx2word = []
        embeddingList = []
        with open("data/glove.6B.100d.txt") as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.split()
                self.idx2word.append(line[0])
                embeddingList.append(line[1:])
        self.embedding = np.concatenate(np.array(embeddingList), axis=0).reshape(-1, 100)

if __name__ == '__main__':
    wd = words()
    with open("data/preloaded.md", "w") as f:
        pickle.dumps(wd)


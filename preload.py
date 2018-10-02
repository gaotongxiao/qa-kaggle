import pdb
import pickle
import numpy as np
from collections import Counter

class words:
    def __init__(self):
        self.idx2word = {0: 'UNK', 1: '<START>', 2: '<EOS>'}
        self.embedding = np.random.randn(400003, 100)
        with open("data/glove.6B.100d.txt") as f:
            a = 3
            while True:
                line = f.readline()
                if not line: break
                line = line.split()
                self.idx2word[a] = line[0]
                self.embedding[a, :] = np.array(line[1:])
                a += 1
        self.word2idx = {v: k for k, v in self.idx2word.items()}

if __name__ == '__main__':
    wd = words()
    pickle.dump(wd, open("data/preloaded.md", "wb"), True)
    test = pickle.load(open("data/preloaded.md", "rb"))


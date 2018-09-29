import numpy as np
import nltk
import csv
import string
import pickle
from preload import words
import h5py

wd = pickle.load(open("data/preloaded.md", "rb"))
word2idx = wd.word2idx
h5 = h5py.File("data/qa.hdf5", "w")
with open('data/train.csv', 'r') as f:
    cf = csv.reader(f, delimiter=',', quotechar='"')
    lema = nltk.WordNetLemmatizer()
    table = str.maketrans({key: None for key in string.punctuation})
    first_line = True
    batch_q1 = []
    batch_q2 = []
    batch_dup = []
    for row in cf:
        if first_line:
            first_line = False
            continue
        q1, q2, is_duplicate = row[3], row[4], row[5]
        batch_dup.append(int(is_duplicate))
        q1 = q1.lower().translate(table)
        newq = []
        q2 = q2.lower().translate(table)
        for word in nltk.word_tokenize(q1):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        newq.append(word2idx['<EOS>'])
        batch_q1.append(newq)
        newq = []
        for word in nltk.word_tokenize(q2):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        batch_q2.append(newq)
    max_len = 0
    for b in batch_q1 + batch_q2:
        max_len = max(max_len, len(b))
with open('data/test.csv', 'r') as f:
    cf = csv.reader(f, delimiter=',', quotechar='"')
    lema = nltk.WordNetLemmatizer()
    table = str.maketrans({key: None for key in string.punctuation})
    first_line = True
    test_batch_q1 = []
    test_batch_q2 = []
    for row in cf:
        if first_line:
            first_line = False
            continue
        q1, q2 = row[1], row[2]
        q1 = q1.lower().translate(table)
        newq = []
        q2 = q2.lower().translate(table)
        for word in nltk.word_tokenize(q1):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        newq.append(word2idx['<EOS>'])
        test_batch_q1.append(newq)
        newq = []
        for word in nltk.word_tokenize(q2):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        test_batch_q2.append(newq)
    max_len = 0
    for b in test_batch_q1 + test_batch_q2:
        max_len = max(max_len, len(b))

for i in range(len(batch_q1)):
    batch_q1[i] += [word2idx['<EOS>'] for _ in range(max_len - len(batch_q1[i]))]
for i in range(len(batch_q2)):
    batch_q2[i] += [word2idx['<EOS>'] for _ in range(max_len - len(batch_q2[i]))]
newbatch = list(zip(batch_q1, batch_q2))
newbatch = np.array(newbatch, dtype=int)
batch_dup = np.array(batch_dup, dtype=int)
train = h5.create_group("train")
train.create_dataset("questions", data=newbatch)
train.create_dataset("is_dup", data=batch_dup)
h5.create_dataset("max_len", data=max_len)

for i in range(len(test_batch_q1)):
    test_batch_q1[i] += [word2idx['<EOS>'] for _ in range(max_len - len(test_batch_q1[i]))]
for i in range(len(test_batch_q2)):
    test_batch_q2[i] += [word2idx['<EOS>'] for _ in range(max_len - len(test_batch_q2[i]))]
newbatch = list(zip(test_batch_q1, test_batch_q2))
newbatch = np.array(newbatch, dtype=int)
test = h5.create_group("test")
test.create_dataset("questions", data=newbatch)


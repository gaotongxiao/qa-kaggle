import numpy as np
import nltk
import csv
import string
import pickle
from preload import words
import h5py
from collections import Counter
import pickle

wd = pickle.load(open("data/preloaded.md", "rb"))
word2idx = wd.word2idx
tf = Counter()

with open('data/train.csv', 'r') as f:
    cf = csv.reader(f, delimiter=',', quotechar='"')
    lema = nltk.WordNetLemmatizer()
    table = str.maketrans({key: None for key in string.punctuation})
    first_line = True
    batch_q1 = []
    batch_q2 = []
    batch_dup = []
    i = 0
    newq = []
    for row in cf:
        if first_line:
            first_line = False
            continue
        q1, q2, is_duplicate = row[3], row[4], row[5]
        batch_dup.append(int(is_duplicate))
        q1 = q1.lower().translate(table)
        q2 = q2.lower().translate(table)
        for word in nltk.word_tokenize(q1):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        newq.append(word2idx['<EOS>'])
        for word in nltk.word_tokenize(q2):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        newq.append(word2idx['<EOS>'])
        i += 1
        if i == 10000:
            tf.update(newq)
            i = 0
            newq = []
with open('data/test.csv', 'r') as f:
    cf = csv.reader(f, delimiter=',', quotechar='"')
    lema = nltk.WordNetLemmatizer()
    table = str.maketrans({key: None for key in string.punctuation})
    first_line = True
    i = 0
    newq = []
    for row in cf:
        if first_line:
            first_line = False
            continue
        q1, q2 = row[1], row[2]
        q1 = q1.lower().translate(table)
        q2 = q2.lower().translate(table)
        for word in nltk.word_tokenize(q1):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        newq.append(word2idx['<EOS>'])
        for word in nltk.word_tokenize(q2):
            w = lema.lemmatize(word)
            try:
                newq.append(word2idx[w])
            except:
                newq.append(word2idx['UNK'])
        newq.append(word2idx['<EOS>'])
        i += 1
        if i == 10000:
            tf.update(newq)
            i = 0
            newq = []

pickle.dump(tf, open("data/tf.md", "wb"))

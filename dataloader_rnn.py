import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from preload import words
import h5py
from torch.utils.data.dataloader import DataLoader
from dataloader import QDataset

wd = pickle.load(open('data/preloaded.md','rb'), encoding='latin1')

def load_data(num_workers=1, batch_size=10):

	f = h5py.File('data/qa.hdf5','r')
	ques_train = f['train']['questions']
	dup_train = f['train']['is_dup']
	ques_test = f['test']['questions']
	embeded_ques_train = embed_data(ques_train)
	embeded_ques_test = embed_data(ques_test)

	train_set = QDataset(embeded_ques_train, dup_train)
	test_set = QDataset(embeded_ques_test)
	training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=False)
	test_data_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=False)

	seq_max_len = f['max_len']
	return training_data_loader, testing_data_loader, seq_max_len.value


def embed_data(dataset):
	embeded_data = F.embedding(torch.LongTensor(dataset), torch.Tensor(wd.embedding))
	return embeded_data
	

if __name__ == '__main__':
	training_data_loader, testing_data_loader, seq_max_len = load_data()
	for batch in training_data_loader:
		print(batch)
		break
	for batch in testing_data_loader:
		print(batch)
		break
	

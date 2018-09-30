import h5py
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import transformer.Constants
import numpy as np
import transformer.Constants as Constants

class QDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor=None):
        if target_tensor is not None:
            assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        if self.target_tensor is None:
            return self.data_tensor[index]
        else:
            return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]

def load_data(num_workers=1, batch_size=10):
    def collate_fn_train(data):
        data = list(zip(*data))
        ques = np.array(data[0]).astype(int)
        is_dup = np.array(data[1]).astype(int)
        batch_size, _, max_len = ques.shape
        pos = (np.arange(max_len) + 1)[np.newaxis, :].repeat(2 * batch_size, 0).reshape(-1, 2, max_len)
        pos = np.multiply(pos, np.array(ques != Constants.EOS))
        return ques, pos, is_dup
    def collate_fn_test(data):
        data = list(zip(*data))
        ques = np.array(data[0]).astype(int)
        ids = np.array(data[1]).astype(int)
        batch_size, _, max_len = ques.shape
        pos = (np.arange(max_len) + 1)[np.newaxis, :].repeat(2 * batch_size, 0).reshape(-1, 2, max_len)
        pos = np.multiply(pos, np.array(ques != Constants.EOS))
        return ques, pos, ids

    f = h5py.File("data/qa.hdf5", 'r')
    ques_train = f['train']['questions']
    dup_train = f['train']["is_dup"]
    ques_test = f['test']["questions"]
    ids_test = f['test']["ids"]
    train_set = QDataset(ques_train, dup_train)
    test_set = QDataset(ques_test, ids_test)
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                      shuffle=True, collate_fn=collate_fn_train)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                     shuffle=False, collate_fn=collate_fn_test)
    seq_max_len = f['max_len']
    return training_data_loader, testing_data_loader, seq_max_len.value

if __name__ == '__main__':
    training_data_loader, testing_data_loader, seq_max_len = load_data()
    for batch in training_data_loader:
        print(batch)
        break
    for batch in testing_data_loader:
        print(batch)
        break

import h5py
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.DataLoader as DataLoader

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
        # print(index)
        if self.target_tensor is None:
            return self.data_tensor[index]
        else:
            return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]

def load_data():
    f = h5py.File("data/qa.hdf5", 'r')
    ques_train = f['train']['questions']
    dup_train = f['train']["is_dup"]
    ques_test = f['test']["questions"]
    train_set = QDataset(ques_train, dup_train)
    test_set = QDataset(ques_test)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, pin_memory=True,
                                     shuffle=False)
    return training_data_loader, testing_data_loader
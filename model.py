import pickle
import torch
import torch.nn as nn
from transformer.Models import Encoder
from dataloader import load_data
from preload import words

class Model(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=100, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            word_vec=None):
        super().__init__()
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, word_vec=word_vec)
    def forward(self, src_seq, src_pos):
        return self.encoder(src_seq, src_pos)


if __name__ == '__main__':
    training_data_loader, testing_data_loader, seq_max_len = load_data(2, 16)

    wd = pickle.load(open("data/preloaded.md", "rb"))
    embed = torch.from_numpy(wd.embedding).cuda()
    model = Model(400003, seq_max_len, word_vec=embed).cuda()
    a = self.encoder()
    for batch in training_data_loader:
        break
        print(batch)
    print("hello")


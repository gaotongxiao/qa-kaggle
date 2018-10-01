import numpy as np
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Encoder
from dataloader import load_data
from preload import words
from transformer import Constants

class Model(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=100, d_model=100, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            word_vec=None, tf_vec=None):
        super().__init__()
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, word_vec=word_vec)
        self.tf_embed = tf_vec
    def forward(self, ques, pos):
        ques = ques.view(-1, seq_max_len)
        q = ques[0::2]
        Q = ques[1::2]
        mask = q.eq(Constants.EOS)
        pos = pos.view(-1, seq_max_len)
        hidden = self.encoder(ques, pos)[0]
        q_hidden = hidden[0::2]
        Q_hidden = hidden[1::2]
        sim = torch.bmm(q_hidden, Q_hidden.transpose(-1, -2))
        g = sim.max(-1)[0]
        g.masked_fill_(mask, 0)
        tf_q = F.embedding(q, tf_embed).squeeze(-1).float()
        tf_q.masked_fill_(mask, 0)
        tf_q /= torch.sum(tf_q, -1, keepdim=True)
        d_mask = torch.sum((q.unsqueeze(-1) - Q.unsqueeze(-2)).eq(0), -1).gt(0)
        tf_q.masked_fill_(d_mask, 1)
        d = tf_q
        g = torch.sum(d * g, -1)
        return g


if __name__ == '__main__':
    cuda = True
    training_data_loader, testing_data_loader, seq_max_len = load_data(1, 64)
    wd = pickle.load(open("data/preloaded.md", "rb"), encoding='latin1')
    embed = torch.from_numpy(wd.embedding).type(torch.FloatTensor)
    tf_embed = torch.from_numpy(pickle.load(open("data/tf_np.md", "rb"))).type(torch.cuda.LongTensor)
    tf_embed.requires_grad = False
    model = Model(400003, seq_max_len, word_vec=embed, tf_vec=tf_embed).cuda()
    device = torch.device('cuda' if cuda else 'cpu')
    Loss = nn.BCEWithLogitsLoss()
    # Loss = nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),betas=(0.9, 0.98), eps=1e-09)

    model.train()
    epochs = 200

    for e in range(epochs):
        avg_loss = []
        for batch in training_data_loader:
            ques, pos, is_dup = map(lambda x: torch.from_numpy(x).to(device), batch)
            is_dup = is_dup.float()
            score = model(ques, pos)
            loss = Loss(score, is_dup.float())
            avg_loss.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if e % 10 == 0:
            print(avg_loss)
            print(np.mean(avg_loss))
            avg_loss = []


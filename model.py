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

def c(data):
    if torch.isnan(data).sum(-1).gt(0)[0]:
        pdb.set_trace()
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
        g.register_hook(c)
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
    USE_CUDA = torch.cuda.is_available()
    BATCH_SIZE = 64
    th = torch.cuda if USE_CUDA else torch

    training_data_loader, testing_data_loader, seq_max_len = load_data(4, BATCH_SIZE)
    wd = pickle.load(open("data/preloaded.md", "rb"), encoding='latin1')
    embed = torch.from_numpy(wd.embedding).type(torch.FloatTensor)
    tf_embed = torch.from_numpy(pickle.load(open("data/tf_np.md", "rb"))).type(th.LongTensor)
    tf_embed.requires_grad = False
    model = Model(400003, seq_max_len, word_vec=embed, tf_vec=tf_embed).cuda()
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    Loss = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),betas=(0.9, 0.98), eps=1e-09)
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),1e-03)

    model.train()
    epochs = 200

    for e in range(epochs):
        avg_loss = []
        for batch in training_data_loader:
            # preprocess
            ques, is_dup = map(lambda x: x.to(device), batch)
            _, _, max_len = ques.shape
            pos = (torch.arange(max_len) + 1).type(th.LongTensor).unsqueeze(0).repeat(2 * BATCH_SIZE, 1).view(-1, 2, max_len)
            pos = pos * (ques != Constants.EOS).long()
            is_dup = is_dup.float()
            score = model(ques, pos)
            loss = Loss(score, is_dup.float())
            print(loss.data)
            avg_loss.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if e % 5 == 0:
            print('saving')
            torch.save(model.state_dict(), 'checkpoints/model_%d.pkl' % e)
        
        print(np.mean(avg_loss))
        avg_loss = []


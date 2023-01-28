import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import *


START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_crf(nn.Module):
    def __init__(self, config):
        super(BiLSTM_crf, self).__init__()
        self.config = config
        self.device = config.device

        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.tag2id = config.tag2id
        self.tagset_size = len(self.tag2id)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[:, self.tag2id[START_TAG]] = -1000.
        self.transitions.data[self.tag2id[STOP_TAG], :] = -1000.

        # map input tokens to unique embedding(vector)
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        # LSTM is a variant of Recurrent Neural Network(RNN)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # linear layer, predict the probability of each tag
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

    def real_path_score(self, logits, label):
        score = torch.zeros(1).to(self.config.device)
        label = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), torch.tensor(label)])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag2id[STOP_TAG]]
        return score

    def total_score(self, logits, label):
        emit = []
        emit = torch.tensor(emit).to(self.config.device)
        previous = torch.full((1, self.tagset_size), 0).to(self.config.device)
        for index in range(len(logits)):
            previous = previous.expand(self.tagset_size, self.tagset_size).t()
            emit = logits[index].view(1, -1).expand(self.tagset_size, self.tagset_size)
            scores = previous + emit + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag2id[STOP_TAG]]
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sent, labels, lengths, mask):
        embedded = self.emb(sent)
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embedded)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # lstm_out: [batch_size, max_len, hidden_dim]
        logits = self.hidden2tag(lstm_out)  # logits: [batch_size, max_len, tagset_size]

        real_path_score = torch.zeros(1).to(self.config.device)
        total_score = torch.zeros(1).to(self.config.device)
        for logit, label, length in zip(logits, labels, lengths):
            logit = logit[:length]
            real_path_score += self.real_path_score(logit, label)
            total_score += self.total_score(logit, label)
        return total_score - real_path_score

    def forward(self, sent, labels, lengths, mask):
        embedded = self.emb(sent)
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embedded)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # lstm_out: [batch_size, max_len, hidden_dim]
        logits = self.hidden2tag(lstm_out)  # logits: [batch_size, max_len, tagset_size]

        scores = []
        paths = []
        for logit, length in zip(logits, lengths):
            logit = logit[:length]
            score, path = self.viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def viterbi_decode(self, logits):
        trellis = torch.zeros(logits.size()).to(self.config.device)
        backpointers = torch.zeros(logits.size(), dtype=torch.long).to(self.config.device)
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.cpu().numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        viterbi_score = torch.tensor(viterbi_score).to(self.config.device)
        viterbi = torch.tensor(viterbi).to(self.config.device)
        return viterbi_score, viterbi

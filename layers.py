"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class CharacterConvNet(nn.Module):
    """Character-level convolutional embedder."""
    def __init__(self, char_vectors, num_filters):
        super(CharacterConvNet, self).__init__()
        self.num_filters = num_filters
        self.embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.conv_net = nn.Conv1d(in_channels=char_vectors.size(1), out_channels=num_filters, kernel_size=5, padding=2)

    def forward(self, chars):
        emb = self.embed(chars)
        batch_size, sent_len, word_len, char_emb_dim = emb.shape
        emb = emb.permute(0, 1, 3, 2).contiguous()
        emb = emb.view(batch_size * sent_len, char_emb_dim, word_len)
        emb = self.conv_net(emb)
        emb = emb.view(batch_size, sent_len, self.num_filters, word_len)
        emb = torch.max(emb, dim=-1)[0]

        return emb


class Embedding(nn.Module):
    """Embedding layer used by BiDAF with the character-level component.

    Word-level and character-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob, use_chars=False, char_vectors=None, num_filters=100):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed_words = nn.Embedding.from_pretrained(word_vectors, freeze=False)
        self.embed_chars = CharacterConvNet(char_vectors, num_filters) if use_chars else None
        embedding_dim = word_vectors.size(1) + num_filters if use_chars else word_vectors.size(1)
        self.proj = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, words, chars):
        emb = self.embed_words(words)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        if self.embed_chars is not None:
            emb_chars = self.embed_chars(chars)
            emb_chars = F.dropout(emb_chars, self.drop_prob, self.training)
            emb = torch.cat((emb, emb_chars), dim=-1)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFAttentionMirrored(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttentionMirrored, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a_c = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b_c = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x_c = torch.cat([c, a_c, c * a_c, c * b_c], dim=2)  # (bs, c_len, 4 * hid_size)

        a_q = torch.bmm(s2.transpose(1, 2), c)

        b_q = torch.bmm(torch.bmm(s2.transpose(1, 2), s1), q)

        x_q = torch.cat([q, a_q, q * a_q, q * b_q], dim=2)

        return x_c, x_q

    def get_similarity_matrix(self, c, q):
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class SQuADOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, hidden_size_2, drop_prob):
        super(SQuADOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size_2, 1)
        self.att_linear_1_start = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1_start = nn.Linear(2 * hidden_size_2, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size_2,
                              hidden_size=hidden_size_2,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size_2, 1)
        self.att_linear_2_start = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2_start = nn.Linear(2 * hidden_size_2, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1_start = self.att_linear_1(att[:, 0].unsqueeze(1)) + self.mod_linear_1(mod[:, 0].unsqueeze(1))
        logits_1 = self.att_linear_1(att[:, 1:]) + self.mod_linear_1(mod[:, 1:])
        logits_1 = torch.cat((logits_1_start, logits_1), dim=1)

        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2_start = self.att_linear_2(att[:, 0].unsqueeze(1)) + self.mod_linear_2(mod_2[:, 0].unsqueeze(1))
        logits_2 = self.att_linear_2(att[:, 1:]) + self.mod_linear_2(mod_2[:, 1:])
        logits_2 = torch.cat((logits_2_start, logits_2), dim=1)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class GTOutput(nn.Module):
    def __init__(self, hidden_size):
        super(GTOutput, self).__init__()
        self.att_linear = nn.Linear(8 * hidden_size, 1)
        self.mod_linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, gap_indices):
        batch_size = mod.shape[0]
        index_all = torch.arange(batch_size).unsqueeze(-1)
        att_gaps = att[index_all, gap_indices]
        mod_gaps = mod[index_all, gap_indices]

        logits = self.att_linear(att_gaps) + self.mod_linear(mod_gaps)

        return logits.squeeze()


class GTOutputWindowPooling(nn.Module):
    def __init__(self, hidden_size):
        super(GTOutputWindowPooling, self).__init__()
        self.att_linear = nn.Linear(3 * 8 * hidden_size, 1)
        self.mod_linear = nn.Linear(3 * 2 * hidden_size, 1)
        self.att_linear_start = nn.Linear(3 * 8 * hidden_size, 1)
        self.mod_linear_start = nn.Linear(3 * 2 * hidden_size, 1)

    def forward(self, att, mod, gap_indices, mask):
        batch_size, seq_len, _ = mod.shape
        device = gap_indices.device
        index_all = torch.arange(batch_size).unsqueeze(-1)
        att_gaps = att[index_all, gap_indices]
        mod_gaps = mod[index_all, gap_indices]

        pools_mod = []
        pools_att = []
        for num, gap_id in enumerate(torch.split(gap_indices, split_size_or_sections=1, dim=1)):
            if num == 0:
                max_pool_mod, _ = torch.max(mod, dim=1, keepdim=True)
                avg_pool_mod = torch.sum(mod, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(mask, dim=1, keepdim=True), dim=2).type(torch.float32)
                max_pool_att, _ = torch.max(att, dim=1, keepdim=True)
                avg_pool_att = torch.sum(att, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(mask, dim=1, keepdim=True), dim=2).type(torch.float32)

            else:
                indices = torch.arange(seq_len).view(1, -1).to(device)
                window_mask = (indices >= (gap_id - 12)) * (indices <= (gap_id + 12)) * mask
                mod_window = mod * window_mask.unsqueeze(2).type(torch.float32)
                max_pool_mod, _ = torch.max(mod_window, dim=1, keepdim=True)
                avg_pool_mod = torch.sum(mod_window, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(window_mask, dim=1, keepdim=True), dim=2).type(torch.float32)
                att_window = att * window_mask.unsqueeze(2).type(torch.float32)
                max_pool_att, _ = torch.max(att_window, dim=1, keepdim=True)
                avg_pool_att = torch.sum(att_window, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(window_mask, dim=1, keepdim=True), dim=2).type(torch.float32)
            pools_mod.append(torch.cat((max_pool_mod, avg_pool_mod), dim=-1))
            pools_att.append(torch.cat((max_pool_att, avg_pool_att), dim=-1))

        pools_mod = torch.cat(pools_mod, dim=1)
        pools_att = torch.cat(pools_att, dim=1)

        mod_gaps = torch.cat((mod_gaps, pools_mod), dim=-1)
        att_gaps = torch.cat((att_gaps, pools_att), dim=-1)

        logits_start = self.att_linear_start(att_gaps[:, 0].unsqueeze(1)) + self.mod_linear_start(mod_gaps[:, 0].unsqueeze(1))
        logits = self.att_linear(att_gaps[:, 1:]) + self.mod_linear(mod_gaps[:, 1:])
        logits = torch.cat((logits_start, logits), dim=1)

        return logits.squeeze()


class GTOutputDoubleWindowPooling(nn.Module):
    def __init__(self, hidden_size, hidden_size_2):
        super(GTOutputDoubleWindowPooling, self).__init__()
        self.att_linear = nn.Linear(5 * 8 * hidden_size, 1)
        self.mod_linear = nn.Linear(5 * 2 * hidden_size_2, 1)
        self.att_linear_start = nn.Linear(5 * 8 * hidden_size, 1)
        self.mod_linear_start = nn.Linear(5 * 2 * hidden_size_2, 1)

    def forward(self, att, mod, gap_indices, mask):
        batch_size, seq_len, _ = mod.shape
        device = gap_indices.device
        index_all = torch.arange(batch_size).unsqueeze(-1)
        att_gaps = att[index_all, gap_indices]
        mod_gaps = mod[index_all, gap_indices]

        pools_mod_1 = []
        pools_att_1 = []
        pools_mod_2 = []
        pools_att_2 = []
        for num, gap_id in enumerate(torch.split(gap_indices, split_size_or_sections=1, dim=1)):
            if num == 0:
                max_pool_mod_1, _ = torch.max(mod[:, 1:], dim=1, keepdim=True)
                avg_pool_mod_1 = torch.sum(mod[:, 1:], dim=1, keepdim=True) / torch.unsqueeze(torch.sum(mask, dim=1, keepdim=True) - 1, dim=2).type(torch.float32)
                max_pool_att_1, _ = torch.max(att[:, 1:], dim=1, keepdim=True)
                avg_pool_att_1 = torch.sum(att[:, 1:], dim=1, keepdim=True) / torch.unsqueeze(torch.sum(mask, dim=1, keepdim=True) - 1, dim=2).type(torch.float32)

                max_pool_mod_2, _ = torch.max(mod_gaps[:, 1:], dim=1, keepdim=True)
                avg_pool_mod_2 = torch.sum(mod_gaps[:, 1:], dim=1, keepdim=True) / 4
                max_pool_att_2, _ = torch.max(att_gaps[:, 1:], dim=1, keepdim=True)
                avg_pool_att_2 = torch.sum(att_gaps[:, 1:], dim=1, keepdim=True) / 4

            else:
                indices = torch.arange(seq_len).view(1, -1).to(device)
                window_mask_1 = (indices >= (gap_id - 12)) * (indices < gap_id) * mask
                window_mask_2 = (indices > gap_id) * (indices <= (gap_id + 12)) * mask

                mod_window_1 = mod * window_mask_1.unsqueeze(2).type(torch.float32)
                max_pool_mod_1, _ = torch.max(mod_window_1, dim=1, keepdim=True)
                avg_pool_mod_1 = torch.sum(mod_window_1, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(window_mask_1, dim=1, keepdim=True), dim=2).type(torch.float32)
                att_window_1 = att * window_mask_1.unsqueeze(2).type(torch.float32)
                max_pool_att_1, _ = torch.max(att_window_1, dim=1, keepdim=True)
                avg_pool_att_1 = torch.sum(att_window_1, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(window_mask_1, dim=1, keepdim=True), dim=2).type(torch.float32)

                mod_window_2 = mod * window_mask_2.unsqueeze(2).type(torch.float32)
                max_pool_mod_2, _ = torch.max(mod_window_2, dim=1, keepdim=True)
                avg_pool_mod_2 = torch.sum(mod_window_2, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(window_mask_2, dim=1, keepdim=True), dim=2).type(torch.float32)
                att_window_2 = att * window_mask_2.unsqueeze(2).type(torch.float32)
                max_pool_att_2, _ = torch.max(att_window_2, dim=1, keepdim=True)
                avg_pool_att_2 = torch.sum(att_window_2, dim=1, keepdim=True) / torch.unsqueeze(torch.sum(window_mask_2, dim=1, keepdim=True), dim=2).type(torch.float32)

            pools_mod_1.append(torch.cat((max_pool_mod_1, avg_pool_mod_1), dim=-1))
            pools_att_1.append(torch.cat((max_pool_att_1, avg_pool_att_1), dim=-1))
            pools_mod_2.append(torch.cat((max_pool_mod_2, avg_pool_mod_2), dim=-1))
            pools_att_2.append(torch.cat((max_pool_att_2, avg_pool_att_2), dim=-1))

        pools_mod_1 = torch.cat(pools_mod_1, dim=1)
        pools_att_1 = torch.cat(pools_att_1, dim=1)
        pools_mod_2 = torch.cat(pools_mod_2, dim=1)
        pools_att_2 = torch.cat(pools_att_2, dim=1)

        mod_gaps = torch.cat((mod_gaps, pools_mod_1, pools_mod_2), dim=-1)
        att_gaps = torch.cat((att_gaps, pools_att_1, pools_att_2), dim=-1)

        logits_start = self.att_linear_start(att_gaps[:, 0].unsqueeze(1)) + self.mod_linear_start(mod_gaps[:, 0].unsqueeze(1))
        logits = self.att_linear(att_gaps[:, 1:]) + self.mod_linear(mod_gaps[:, 1:])
        logits = torch.cat((logits_start, logits), dim=1)

        return logits.squeeze()


class MLMNSPOutput(nn.Module):
    def __init__(self, hidden_size, hidden_size_2, vocab_size):
        super(MLMNSPOutput, self).__init__()
        self.mod_linear = nn.Linear(2 * hidden_size_2, vocab_size)
        self.mod_linear_start = nn.Linear(3 * 2 * hidden_size_2, 1)
        self.att_linear_start = nn.Linear(3 * 8 * hidden_size, 1)

    def forward(self, c_att, c_mod, q_att, q_mod, mask_1, mask_2, texts_1_words_unmasked, texts_2_words_unmasked):
        start_mod = torch.cat((c_mod[:, 0], q_mod[:, 0], c_mod[:, 0] * q_mod[:, 0]), dim=-1)
        start_att = torch.cat((c_att[:, 0], q_att[:, 0], c_att[:, 0] * q_att[:, 0]), dim=-1)
        NSP_logits = self.mod_linear_start(start_mod) + self.att_linear_start(start_att)

        MLM_mod_1 = c_mod[mask_1]
        MLM_mod_2 = q_mod[mask_2]

        MLM_logits_1 = self.mod_linear(MLM_mod_1)
        MLM_logits_2 = self.mod_linear(MLM_mod_2)

        masked_words_1 = texts_1_words_unmasked[mask_1]
        masked_words_2 = texts_2_words_unmasked[mask_2]

        MLM_logits = torch.cat((MLM_logits_1, MLM_logits_2), dim=0)
        masked_words = torch.cat((masked_words_1, masked_words_2), dim=0)

        return MLM_logits.squeeze(), masked_words, NSP_logits.squeeze()

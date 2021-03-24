import torch
from numpy import sqrt
import warnings

'''
Multi-Headed Attention Module as defined by Alfredo Canziani
https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-3/ 
'''

class MultiHeadedAttention(torch.nn.Module):
    # using d_input changes from self to cross-attention
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq = d_xk = d_xv = d_input

        # embedding dimension is a multiple of num_heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model//self.num_heads

        #These are still of dimension d_model
        self.W_q = torch.nn.Linear(d_xq, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_xq, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_xq, d_model, bias=False)
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = torch.nn.Linear(d_model, d_model)

    def scaled_dot_product_attn(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        #scaling by d_k to prevent saturation
        Q = Q / sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(2, 3))
        A = torch.nn.Softmax(dim=-1)(scores)

        # Get weighted average
        # [vs, n_heads, q_length, dim_per_head]
        H = torch.matmul(A,V)
        return H, A

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

    def group_heads(self, x, batch_size):
        return x.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads *self.d_k)

    def forward(self, X):
        if len(X.size()) > 3:
            warnings.warn('Dimensions Do Not Match: Attempting to flatten')
            X = X.view(X.size(0), X.size(1), -1)
        X_q = X_k = X_v = X
        batch_size, seq_len, dim = X_q.size()

        Q = self.split_heads(self.W_q(X_q), batch_size)
        K = self.split_heads(self.W_k(X_k), batch_size)
        V = self.split_heads(self.W_v(X_v), batch_size)

        H_cat, A = self.scaled_dot_product_attn(Q,K,V)
        H_cat = self.group_heads(H_cat, batch_size)
        H = self.W_h(H_cat)
        return H
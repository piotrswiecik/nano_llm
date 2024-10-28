"""Self-attention module"""

import torch


class SelfAttention(torch.nn.Module):
    # TODO: batch support
    def __init__(self, d_in: int, d_out: int):
        """
        Args:
            d_in (int): The input dimension (size of token embedding).
            d_out (int): The output dimension (size of context vector).
        """
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_out).
        """
        keys = x @ self.W_key # (bs, seq_len, d_in) x (d_in, d_out) -> (bs, seq_len, d_out)
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # (bs, seq_len, d_out) x (bs, seq_len, d_out).T -> (bs, seq_len, seq_len)
        attn_weights = torch.nn.functional.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        ctx = attn_weights @ values # (bs, seq_len, seq_len) x (bs, seq_len, d_out) -> (bs, seq_len, d_out)
        return ctx
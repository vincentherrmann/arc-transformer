import torch
import math


class MultiheadAttention(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 num_heads,
                 prior_dim=0,
                 bias=False,
                 dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.use_bias = bias

        self.query_transform = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                               bias=bias)
        if prior_dim > 0:
            self.prior_transform = torch.nn.Linear(in_features=feature_dim, out_features=prior_dim * num_heads,
                                                   bias=bias)
        else:
            self.prior_transform = None
        self.key_transform = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                             bias=bias)
        self.value_transform = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                               bias=bias)
        self.output_projection = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                                 bias=bias)

        self.dropout = torch.nn.Dropout(dropout)
        self.logit_scaling = 1 / math.sqrt(feature_dim / self.num_heads + prior_dim)

        self._reset_parameters()

    def forward(self, query_source, key_source, value_source=None, attention_prior=None, attention_mask=None):
        # query_source: batch x target_seq x query_dim
        # key_source: batch x source_seq x key_dim
        # value_source: batch x source_seq x value_dim (if None, then key_source will be used)
        # attention_prior: batch x target_seq x source_seq x prior_dim
        # attention_mask: target_seq x source_seq

        if value_source is None:
            value_source = key_source

        n, t, e = query_source.shape
        s = key_source.shape[1]
        h = self.num_heads
        eh = e // self.num_heads

        query = self.query_transform(query_source).view(n, t, h, eh)  # n x t x h x eh
        key = self.key_transform(key_source).view(n, s, h, eh)  # n x s x h x eh
        value = self.value_transform(value_source).view(n, s, h, eh)  # n x s x h x eh

        query = query.permute(0, 2, 1, 3).reshape(n * h, t, eh)  # n*h x t x eh
        key = key.permute(0, 2, 3, 1).reshape(n * h, eh, s)  # n*h x eh x s
        value = value.permute(0, 2, 1, 3).reshape(n * h, s, eh)

        content_based_logits = torch.bmm(query, key)  # n*h x t x s

        if attention_prior is not None:
            p = attention_prior.shape[3]
            prior_query = self.prior_transform(query_source).view(t, n, h, p)
            prior_query = prior_query.permute(1, 2, 0, 3)  # n x h x t x p
            prior_based_logits = torch.einsum('nhtp,ntsp->nhts', prior_query, attention_prior).reshape(n*h, t, s)
        else:
            prior_based_logits = 0

        logits = content_based_logits + prior_based_logits  # n*h x t x s
        logits *= self.logit_scaling

        if attention_mask is not None:
            logits = logits.view(n, h, t, s)
            logits += attention_mask[:, None, :, :]
            logits = logits.view(n*h, t, s)

        weights = torch.softmax(logits, dim=2)  # n*h x t x s
        weights = self.dropout(weights)
        weighted_value = torch.bmm(weights, value)  # n*h x t x eh
        weighted_value = weighted_value.view(n, h, t, eh).permute(0, 2, 1, 3).reshape(n, t, e)

        output = self.output_projection(weighted_value)
        return output

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query_transform.weight)
        torch.nn.init.xavier_uniform_(self.key_transform.weight)
        torch.nn.init.xavier_uniform_(self.value_transform.weight)
        torch.nn.init.xavier_uniform_(self.output_projection.weight)

        if self.use_bias:
            torch.nn.init.constant_(self.query_transform.bias, 0.)
            torch.nn.init.constant_(self.key_transform.bias, 0.)
            torch.nn.init.constant_(self.value_transform.bias, 0.)
            torch.nn.init.constant_(self.output_projection.bias, 0.)

        if self.prior_transform is not None:
            torch.nn.init.xavier_uniform_(self.prior_transform.weight)
            if self.use_bias:
                torch.nn.init.constant_(self.prior_transform.bias, 0.)
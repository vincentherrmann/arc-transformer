import torch
import torch.nn.functional as F
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


class RelativeMultiheadAttention(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 num_heads,
                 relative_dim=0,
                 bias=False,
                 dropout=0.):
        super().__init__()
        self.feature_dim = feature_dim
        self.relative_dim = relative_dim
        self.num_heads = num_heads
        self.use_bias = bias

        self.query_transform = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                               bias=bias)
        self.key_content_transform = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                                     bias=bias)
        if self.relative_dim > 0:
            self.key_location_transform = torch.nn.Linear(in_features=relative_dim, out_features=feature_dim,
                                                          bias=bias)
        self.value_transform = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                               bias=bias)
        self.output_projection = torch.nn.Linear(in_features=feature_dim, out_features=feature_dim,
                                                 bias=bias)
        self.register_buffer('query_content_bias', torch.rand(1, self.feature_dim))
        self.register_buffer('query_location_bias', torch.rand(1, self.feature_dim))

        self.dropout = torch.nn.Dropout(dropout)
        self.logit_scaling = 1 / math.sqrt(feature_dim / self.num_heads)

        self._reset_parameters()

    def forward(self, query_source, key_source, value_source, attention_mask=None,
                relative_attention_features=None, r_dims=None):
        # query_source: target_seq x batch x emb_dimension
        # key_source: source_seq x batch x emb_dimension
        # value_source: source_seq x batch x emb_dimension (if None, then key_source will be used)
        # attention_mask: t x s
        # relative_attention_features: relative_seq x batch x rel_dimension
        # r_dims: (h, w) for 2d attention

        if value_source is None:
            value_source = key_source

        t, n, e = query_source.shape
        s = key_source.shape[0]
        h = self.num_heads
        eh = e // self.num_heads

        query = self.query_transform(query_source).view(t, n, h, eh)  # t x n x h x eh
        key = self.key_content_transform(key_source).view(s, n, h, eh)  # s x n x h x eh
        value = self.value_transform(value_source).view(s, n, h, eh)  # s x n x h x eh

        # content based attention weights:
        #           t x e               x           e x s
        # query_transform(query_source) x key_content_transform(key_source)

        query = query.permute(1, 2, 0, 3).view(n * h, t, eh)  # n*h x t x eh
        key = key.permute(1, 2, 3, 0).view(n * h, eh, s)  # n*h x eh x s
        value = value.permute(1, 2, 0, 3).view(n * h, s, eh)

        content_based_logits = torch.bmm(query, key)  # n*h x t x s
        content_bias = torch.bmm(self.query_content_bias.repeat(n, 1).view(n*h, 1, eh), key)  # n*h x 1 x s

        logits = content_based_logits + content_bias  # n*h x t x s

        if self.relative_dim > 0:
            r = relative_attention_features.shape[1]
            location_key = self.key_location_transform(relative_attention_features).view(r, n, h, eh)  # r x 1 x h x eh
            #location_key = location_key.repeat([1, n, 1, 1])  # r x n x h x eh
            location_key = location_key.permute(1, 2, 3, 0).view(n * h, eh, -1)  # n*h x eh x r
            unskewed_location_logits = torch.bmm(query, location_key)  # n*h x t x r
            skewed_location_logits = self.skew(unskewed_location_logits, r_dims)  # n*h x t x s
            unskewed_location_bias = torch.bmm(self.query_location_bias.repeat(n, 1).view(n * h, 1, eh),
                                               location_key)  # n*h x 1 x r
            skewed_location_bias = self.skew(unskewed_location_bias.repeat(1, t, 1), r_dims)  # n*h x t x s
            logits += skewed_location_logits + skewed_location_bias  # n*h x t x s

        if attention_mask is not None:
            logits = logits.view(n, h, t, s)
            logits += attention_mask[:, None, :, :]
            logits = logits.view(n*h, t, s)

        weights = torch.softmax(logits, dim=2)  # n*h x t x s
        #self.weight_writer(weights.clone().detach().permute(1, 2, 0).view(t, s, n, h))
        weights = self.dropout(weights)
        weighted_value = torch.bmm(weights, value)  # n*h x t x eh
        weighted_value = weighted_value.view(n, h, t, eh).permute(2, 0, 1, 3).reshape(t, n, e)

        output = self.output_projection(weighted_value)
        return output

    @staticmethod
    def skew(unskewed_logits, diagonal_index):
        '''
        :param unskewed_logits: n x t x r   n: batch, t: target dimension (query), r: relative position
        :param diagonal_index: index of the zero position in the r dimension (the relative position that will be on the
                         main diagonal)
        :return: skewed logits with the shape n x t x p where p = max(diagonal_index + 1, r - diagonal_index)

        unskewed:         skewed
        -2 -1  0  1  2     0  1  2
        -2 -1  0  1  2    -1  0  1  2
        -2 -1  0  1  2    -2 -1  0  1  2
        -2 -1  0  1  2       -2 -1  0  1
        -2 -1  0  1  2          -2 -1  0
        '''

        dev = unskewed_logits.device
        n, t, r = unskewed_logits.shape
        num_past_steps = (r - diagonal_index - 1)
        num_future_steps = diagonal_index
        out_steps = max(num_future_steps, num_past_steps)

        padded_unskewed_logits = torch.cat([unskewed_logits, torch.zeros(n, t, 1).to(dev)], dim=2)
        skewed_logits = padded_unskewed_logits.view(n, -1)[:, diagonal_index:(diagonal_index + t * r)]
        skewed_logits = skewed_logits.view(n, t, r)
        skewed_logits = skewed_logits[:, :, :out_steps]

        # mask invalid positions
        minus_inf = torch.full((t, out_steps), fill_value=float('-inf')).to(dev)
        future_mask = torch.triu(minus_inf, diagonal=num_future_steps+1)
        past_mask = torch.tril(minus_inf, diagonal=-(num_past_steps+1))

        masked_skewed_logits = skewed_logits + future_mask + past_mask
        return masked_skewed_logits

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query_transform.weight)
        torch.nn.init.xavier_uniform_(self.key_content_transform.weight)
        torch.nn.init.xavier_uniform_(self.value_transform.weight)
        torch.nn.init.xavier_uniform_(self.output_projection.weight)

        if self.use_bias:
            torch.nn.init.constant_(self.query_transform.bias, 0.)
            torch.nn.init.constant_(self.key_content_transform.bias, 0.)
            torch.nn.init.constant_(self.value_transform.bias, 0.)
            torch.nn.init.constant_(self.output_projection.bias, 0.)

        if self.relative_dim > 0:
            torch.nn.init.xavier_uniform_(self.key_location_transform.weight)
            if self.use_bias:
                torch.nn.init.constant_(self.key_location_transform.bias, 0.)


class RelativeMultiheadAttention2d(RelativeMultiheadAttention):
    @staticmethod
    def skew(unskewed_logits, size):
        '''
        :param unskewed_logits: n x t x r   n: batch, t: target dimension (query), r: relative position
        :param diagonal_index: index of the zero position in the r dimension (the relative position that will be on the
                         main diagonal)
        :return: skewed logits with the shape n x t x p where p = max(diagonal_index + 1, r - diagonal_index)

        skewed
        -1-1  -1 0  -1 1   0-1  [0 0   0 1]  1-1  [1 0   1 1]
              -1-1  -1 0  -1 1  [0-1   0 0]  0 1  [1-1   1 0]
                    -1-1 [-1 0  -1 1]  0-1   0 0   0 1   1-1
                         [-1-1  -1 0] -1 1   0-1   0 0   0 1

              -1 0  -1 1         0 0   0 1         1 0   1 1
              -1-1  -1 0         0-1   0 0         1-1   1 0
                          -1 0  -1 1         0 0   0 1
                          -1-1  -1 0         0-1   0 0

        skewed
        -1-1  -1 0  -1 1   0-1  [0 0   0 1]  1-1  [1 0   1 1]
        -1-1  -1 0  -1 1  [0-1   0 0]  0 1  [1-1   1 0]
        -1-1 [-1 0  -1 1]  0-1   0 0   0 1   1-1
        [-1-1  -1 0] -1 1   0-1   0 0   0 1
        '''

        dev = unskewed_logits.device
        n, t, r = unskewed_logits.shape
        h, w = size

        padded_unskewed_logits = torch.cat([unskewed_logits, torch.zeros(n, t, 1, dtype=unskewed_logits.dtype).to(dev)],
                                           dim=2)
        skewed_logits = padded_unskewed_logits.view(n, -1)[:, :(t * r)]
        skewed_logits = skewed_logits.view(n, t, r)
        skewed_logits = skewed_logits[:, :, :r]

        # create mask
        th = 2 * h - 1
        tw = 2 * w - 1
        skew = w - 1
        mask_pattern = torch.full((th, tw), fill_value=False, dtype=torch.bool).to(dev)
        mask_pattern[-h:, -w:] = True
        mask_pattern = mask_pattern.view(1, -1).repeat(h, 1)
        mask_pattern = mask_pattern[:, -(r - skew):]
        mask_pattern = F.pad(mask_pattern.reshape(-1), pad=(skew, (h - 1) * skew), value=False).view(h, th, tw)
        mask_pattern = mask_pattern.view(h, 1, th * tw).repeat(1, w, 1).view(1, h * w, th * tw)

        #masked_logits = skewed_logits + mask_pattern
        selected_logits = skewed_logits[:, mask_pattern[0]].view(n, h*w, h*w)
        return selected_logits

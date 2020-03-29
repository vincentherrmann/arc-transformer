import torch
import torch.nn.functional as F
import math

from arc_transformer.preprocessing import PriorCalculation
from arc_transformer.attention_with_priors import MultiheadAttention
from arc_transformer.dataset import task_map, task_flatten


class TransformerLayer(torch.nn.Module):
    def __init__(self, feature_dim, feedforward_dim, self_attention_module, external_attention_module, dropout=0.):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim, feedforward_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(feedforward_dim, feature_dim)

        self.norm1 = torch.nn.LayerNorm(feature_dim)
        self.norm2 = torch.nn.LayerNorm(feature_dim)
        self.norm3 = torch.nn.LayerNorm(feature_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.self_attention = self_attention_module
        self.external_attention = external_attention_module

    def forward(self, target, external, target_prior=None, external_prior=None, target_mask=None, external_mask=None):
        target2 = self.self_attention(target, target, target,
                                      attention_prior=target_prior, attention_mask=target_mask)

        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.external_attention(target, external, external,
                                          attention_prior=external_prior, attention_mask=external_mask)

        target = target + self.dropout2(target2)
        target = self.norm2(target)
        target2 = self.linear2(self.dropout(F.relu(self.linear1(target))))
        target = target + self.dropout3(target2)
        target = self.norm3(target)
        return target


class ArcTransformer(torch.nn.Module):
    def __init__(self, feature_dim, feedforward_dim, dropout=0., num_heads=8,
                 num_pair_features=64, num_task_features=128, num_iterations=4, num_priors=155):
        super().__init__()
        self.num_pair_features = num_pair_features
        self.num_task_features = num_task_features
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations

        s = 1 / math.sqrt(self.feature_dim)
        self.register_buffer("in_embedding", torch.randn(self.feature_dim) * s)
        self.register_buffer("out_embedding", torch.randn(self.feature_dim) * s)
        self.register_buffer("train_embedding", torch.randn(self.feature_dim) * s)
        self.register_buffer("test_embedding", torch.randn(self.feature_dim) * s)
        self.register_buffer("pair_embedding", torch.randn(10, self.feature_dim) * s)
        self.register_buffer("task_embedding", torch.randn(self.feature_dim) * s)
        self.register_buffer("grid_positional_embedding", torch.randn(30, 30, self.feature_dim) * s)
        self.register_buffer("pair_positional_embedding", torch.randn(self.num_pair_features, self.feature_dim) * s)
        self.register_buffer("task_positional_embedding", torch.randn(self.num_task_features, self.feature_dim) * s)

        self.input_layer = torch.nn.Linear(in_features=11, out_features=feature_dim, bias=True)

        # grid layer
        self_attention = MultiheadAttention(feature_dim=feature_dim,
                                            num_heads=num_heads,
                                            prior_dim=num_priors,
                                            dropout=dropout)
        external_attention = MultiheadAttention(feature_dim=feature_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)
        self.grid_layer = TransformerLayer(feature_dim=feature_dim,
                                           feedforward_dim=feedforward_dim,
                                           self_attention_module=self_attention,
                                           external_attention_module=external_attention,
                                           dropout=dropout)

        # pair layer
        self_attention = MultiheadAttention(feature_dim=feature_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        external_attention = MultiheadAttention(feature_dim=feature_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)
        self.pair_layer = TransformerLayer(feature_dim=feature_dim,
                                           feedforward_dim=feedforward_dim,
                                           self_attention_module=self_attention,
                                           external_attention_module=external_attention,
                                           dropout=dropout)

        # task layer
        self_attention = MultiheadAttention(feature_dim=feature_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        external_attention = MultiheadAttention(feature_dim=feature_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)
        self.task_layer = TransformerLayer(feature_dim=feature_dim,
                                           feedforward_dim=feedforward_dim,
                                           self_attention_module=self_attention,
                                           external_attention_module=external_attention,
                                           dropout=dropout)

    def forward(self, task_data, grid_prior):
        num_train_pairs = len(task_data['train'])
        num_test_pairs = len(task_data['test'])
        num_pairs = num_train_pairs + num_test_pairs

        grids = task_flatten(task_data)[None]
        batch_size, _, w, h = grids.shape
        grids_mask = (grids > 0).float().view(batch_size, num_pairs*2, w*h)  # batch x num_pairs*2 x w*h
        grids_target_mask = torch.einsum('nps,npt->npst', grids_mask, grids_mask).view(batch_size*num_pairs*2,
                                                                                       w*h, w*h)
        ext_grids_mask = F.pad(grids_mask.view(batch_size, num_pairs, 2*w*h), pad=(0, self.num_task_features), value=1.)
        pair_external_mask = ext_grids_mask.repeat(1, self.num_pair_features, 1).view(batch_size*num_pairs,
                                                                                      self.num_pair_features,
                                                                                      2*w*h + self.num_task_features)
        grids_pos_embedding = self.grid_positional_embedding[:w, :h, :].reshape(1, w*h, self.feature_dim)

        grids = F.one_hot(grids, num_classes=11)
        grids = grids.view(batch_size*num_pairs*2, w*h, -1)
        grids_mask = grids_mask.view(batch_size*num_pairs*2, w*h, 1)

        grid_prior = task_flatten(grid_prior)

        pair_features = torch.zeros(batch_size * num_pairs, self.num_pair_features, self.feature_dim)
        pair_features += self.pair_positional_embedding[None]
        pair_features[:num_train_pairs] += self.train_embedding[None, None]
        pair_features[num_train_pairs:] += self.test_embedding[None, None]

        task_features = torch.zeros(batch_size, self.num_task_features, self.feature_dim)
        task_features += self.task_positional_embedding[None]

        grids = self.input_layer(grids.float())
        grids += grids_pos_embedding
        grids[:(num_train_pairs*2)] += self.train_embedding[None, None]
        grids[(num_train_pairs*2):] += self.test_embedding[None, None]
        grids = (grids + grids_pos_embedding) * grids_mask

        # add grid positional embedding (10 features)
        # add pair and task feature positional embedding

        for i in range(self.num_iterations):
            # grid layer
            target = grids
            external = pair_features[:, None, :, :].repeat(1, 2, 1, 1).view(grids.shape[0], self.num_pair_features, -1)
            grids = self.grid_layer(target, external, target_prior=grid_prior, target_mask=grids_target_mask)
            grids = grids * grids_mask

            # pair layer
            target = pair_features
            reshaped_grids = grids.view(batch_size*num_pairs, 2, w*h, self.feature_dim)
            reshaped_grids[:, 0] += self.in_embedding
            reshaped_grids[:, 1] += self.out_embedding
            reshaped_grids = reshaped_grids.view(batch_size*num_pairs, 2*w*h, self.feature_dim)
            task_external = (task_features + self.task_embedding).repeat(num_pairs, 1, 1)  # num_pairs, num_task_features, feature_dim
            external = torch.cat([reshaped_grids, task_external], dim=1)  # num_pairs x 2*w*h + num_task_features x feature_dim
            pair_features = self.pair_layer(target, external, external_mask=pair_external_mask)

            # task layer
            target = task_features
            external = (pair_features.view(batch_size, num_pairs, self.num_pair_features, -1) + \
                        self.pair_embedding[None, :num_pairs, None, :]).view(batch_size, -1, self.feature_dim)
            task_features = self.task_layer(target, external)

        return grids, pair_features, task_features

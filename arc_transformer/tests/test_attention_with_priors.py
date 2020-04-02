from arc_transformer.attention_with_priors import RelativeMultiheadAttention
import torch
import torch.nn.functional as F

def test_skew2d():
    h = 5
    w = 7

    th = 2 * h - 1
    tw = 2 * w - 1

    attention = RelativeMultiheadAttention(1, 1)
    logits = torch.arange(th*tw).view(1, 1, -1).repeat(1, h*w, 1)
    skewed_logits = attention.skew2d(logits, h, w)  # n x h*w x th*tw

    h_logits = skewed_logits // tw - h + 1
    w_logits = skewed_logits % tw - w + 1
    skewed_logits_coord = torch.stack([h_logits[0], w_logits[0]], dim=2)

    target_coord = torch.arange(h*w).view(h, w)
    h_coord = target_coord // w
    w_coord = target_coord % w
    target_coord = torch.stack([h_coord, w_coord], dim=2)

    summed_logits = skewed_logits_coord + target_coord.view(h*w, 1, 2)
    unmasked = summed_logits[torch.isfinite(summed_logits)]
    num_errors = (unmasked < 0).int().sum().item()

    assert unmasked.shape[0] == 2*h*h*w*w
    assert num_errors == 0



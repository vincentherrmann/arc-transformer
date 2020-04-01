from arc_transformer.attention_with_priors import RelativeMultiheadAttention
import torch
import torch.nn.functional as F

def test_skew2d():
    h = 3
    w = 4

    th = 2 * h - 1
    tw = 2 * w - 1

    attention = RelativeMultiheadAttention(1, 1)
    logits = torch.arange(th*tw).view(1, 1, -1).repeat(1, h*w, 1)
    #h_logits = logits // tw - h + 1
    #w_logits = logits % tw - w + 1
    #logits_coord = torch.stack([h_logits[0], w_logits[0]], dim=2)
    #zero_index_pos = (2*w-1) * (h-1) + w-1
    skewed_logits = attention.skew2d(logits, 0, h, w)  # n x h*w x th*tw


    h_logits = skewed_logits // tw - h + 1
    w_logits = skewed_logits % tw - w + 1
    skewed_logits_coord = torch.stack([h_logits[0], w_logits[0]], dim=2)

    target_coord = torch.arange(h*w).view(h, w)
    h_coord = target_coord // w
    w_coord = target_coord % w
    target_coord = torch.stack([h_coord, w_coord], dim=2)
    r_target_coord = target_coord.view(h*w, 1, 2).repeat(1, skewed_logits_coord.shape[1], 1)

    summed_logits = skewed_logits_coord + target_coord.view(h*w, 1, 2)
    mask = (summed_logits[:, :, 0] >= 0) * (summed_logits[:, :, 1] >= 0) * (summed_logits[:, :, 0] < h) * (summed_logits[:, :, 1] < w)

    n = 0
    for hi in range(h):
        print("x:", hi)
        for wi in range(w):
            #print("y:", wi)
            l = []
            for i, v in enumerate(mask[hi*w + wi]):
                if v:
                    l.append(n)
                n+=1
            print(l)

    single_mask = torch.zeros((2*h-1), (2*w-1))
    single_mask[-h:, -w:] = 1.
    single_mask = single_mask.view(1, 1, -1).repeat(1, h*w, 1)
    skewed_mask = attention.skew2d(single_mask, 0, h, w)
    pass


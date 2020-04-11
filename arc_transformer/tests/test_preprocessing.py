from unittest import TestCase
from arc_transformer.preprocessing import Preprocessing

import torch
from matplotlib import pyplot as plt

class TestCalculatePriors(TestCase):
    def test_prior_calculation(self):
        p = Preprocessing()

        test_input = torch.LongTensor([[0, 4, 4, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [1, 0, 2, 2, 0],
                                       [0, 0, 0, 2, 0]])
        priors = p(test_input)
        priors = priors.view(4, 5, 4, 5, -1)
        n = test_input.nelement()
        fig, axs = plt.subplots(n, figsize=(8, 16), dpi=50)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig_num = 0
        for i in range(test_input.shape[0]):
            for j in range(test_input.shape[1]):
                axs[fig_num].imshow(priors[i, j, :, :, 30])
                fig_num += 1
        plt.show()
        pass

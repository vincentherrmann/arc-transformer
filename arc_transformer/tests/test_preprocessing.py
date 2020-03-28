from unittest import TestCase
from arc_transformer.preprocessing import PriorCalculation

import torch
from matplotlib import pyplot as plt

class TestCalculatePriors(TestCase):
    def test_prior_calculation(self):
        p = PriorCalculation()

        test_input = torch.LongTensor([[0, 4, 4, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [1, 0, 2, 2, 0],
                                       [0, 0, 0, 2, 0]])
        patterns, colors, objects, extracted_objects = p(test_input)

        n = test_input.nelement()
        fig, axs = plt.subplots(n, figsize=(8, 16), dpi=50)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig_num = 0
        for i in range(test_input.shape[0]):
            for j in range(test_input.shape[1]):
                axs[fig_num].imshow(patterns[i, j, :, :, 30])
                fig_num += 1
        plt.show()
        pass

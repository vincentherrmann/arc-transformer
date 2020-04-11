from unittest import TestCase
from arc_transformer.dataset import ArcDataset, task_map

class TestArcDataset(TestCase):
    def test_get_item(self):
        dataset = ArcDataset("../../../data/training")
        dataset[0]

    def test_constraint_size(self):
        dataset = ArcDataset("../../../data/training", max_size=(30, 30))
        print("length at with max size (30, 30):", len(dataset))

        dataset = ArcDataset("../../../data/training", max_size=(25, 25))
        print("length at with max size (25, 25):", len(dataset))

        dataset = ArcDataset("../../../data/training", max_size=(20, 20))
        print("length at with max size (20, 20):", len(dataset))

        dataset = ArcDataset("../../../data/training", max_size=(15, 15))
        print("length at with max size (15, 15):", len(dataset))

    def test_dataset_sizes(self):
        dataset = ArcDataset('../../../data/training')
        global h_hist, w_hist
        h_hist = [0] * 31
        w_hist = [0] * 31
        def set_max(t):
            h_hist[t.shape[0]] += 1
            w_hist[t.shape[1]] += 1
        for item in dataset:
            task_map(item, set_max)
        print(h_hist)
        print(w_hist)

from unittest import TestCase
from arc_transformer.dataset import ArcDataset

class TestArcDataset(TestCase):
    def test_get_item(self):
        dataset = ArcDataset("/Users/vincentherrmann/Documents/Projekte/abstraction-and-reasoning-challenge/data/training")
        dataset[0]

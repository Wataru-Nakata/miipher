import unittest
import torch
from miipher.model.miipher import Miipher


class TestMiipher(unittest.TestCase):
    def setUp(self) -> None:
        self.miipher = Miipher(512, 256, 1024, 1024, 4, 2)

    def test_miipher(self):
        phone_feature = torch.rand(2, 129, 512)
        speaker_feature = torch.rand(2, 256)
        ssl_feature = torch.rand(2, 121, 1024)
        output = self.miipher.forward(
            phone_feature, speaker_feature, ssl_feature, torch.tensor([121, 121])
        )
        self.assertTrue(output[0].size() == ssl_feature.size())

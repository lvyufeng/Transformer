import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from ddt import ddt, data, unpack
from src.nn import TransformerEncoderLayer

@ddt
class TestTransformerEncoderLayer(unittest.TestCase):
    @data(1, 4, 8)
    def test_transformerencoderlayer_src_mask(self, nhead):
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        src = Tensor(np.random.rand(batch_size, seqlen, d_model), mindspore.float32)
        src_mask = Tensor(np.zeros((seqlen, seqlen)), mindspore.bool_)

        model(src, src_mask=src_mask)
        model.set_train(False)
        model(src, src_mask=src_mask)

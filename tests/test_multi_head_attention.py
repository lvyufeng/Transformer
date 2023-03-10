import unittest
import mindspore
import numpy as np
from ddt import ddt, data
from src.nn import MultiheadAttention
from mindspore import Tensor

@ddt
class TestMultiHeadAttention(unittest.TestCase):
    @data(mindspore.float32, mindspore.float16)
    def test_multihead_attention_pynative(self, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = MultiheadAttention(embed_dim, num_heads).to_float(dtype)
        q = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        k = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        v = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        out = model(q, k, v)
        self.assertEqual(q.shape, out[0].shape)

    @data(mindspore.float32, mindspore.float16)
    def test_multihead_attention_jit(self, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = MultiheadAttention(embed_dim, num_heads).to_float(dtype)
        q = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        k = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        v = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
        @mindspore.jit
        def forward(q, k, v):
            out = model(q, k, v)
            return out
        out = forward(q, k, v)
        self.assertEqual(q.shape, out[0].shape)

    def test_multihead_attention_dtype_batch_first(self):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        for training in (True, False):
            model = MultiheadAttention(embed_dim, num_heads, batch_first=True)
            if not training:
                model = model.set_train(False)
            q = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            k = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            v = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            # fast path currently doesn't support weights
            out = model(q, k, v, need_weights=False)
            self.assertEqual(q.shape, out[0].shape)

    def test_multihead_attention_dtype_batch_first_jit(self):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        model = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        @mindspore.jit
        def forward(q, k, v, need_weights=False):
            out = model(q, k, v, need_weights=need_weights)
            return out
        for training in (True, False):
            if not training:
                model = model.set_train(False)
            q = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            k = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            v = Tensor(np.random.randn(sl, bs, embed_dim), mindspore.float32)
            # fast path currently doesn't support weights
            out = forward(q, k, v)
            self.assertEqual(q.shape, out[0].shape)

    def test_multihead_attn_fast_path_query_and_bias_have_different_dtypes(self):
        mha = MultiheadAttention(4, 4, batch_first=True).set_train(False)
        query = Tensor(np.random.randn(4, 4, 4), mindspore.float32)
        mha(query, query, query)

    def test_multihead_attn_in_proj_bias_none(self):
        mha = MultiheadAttention(2, 2, bias=False)
        query = Tensor(np.random.randn(2, 2, 2), mindspore.float32)
        mha(query, query, query)

    def test_multihead_attn_in_proj_weight_none(self):
        # Setting kdim == vdim == 2 means that vdim != embed_dim
        # will cause the logic to use per-input project weights, thereby
        # forcing self.in_proj_weight = None
        mha = MultiheadAttention(4, 4, vdim=2, kdim=2)
        query = Tensor(np.random.randn(4, 4, 4), mindspore.float32)
        key = Tensor(np.random.randn(4, 4, 2), mindspore.float32)
        mha(query, key, key)

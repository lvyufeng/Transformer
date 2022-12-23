import unittest
from src.nn import MultiheadAttention
from mindspore import ops

class TestMultiHeadAttention(unittest.TestCase):
    def test_multihead_attention_dtype(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = MultiheadAttention(embed_dim, num_heads).cuda().to(dtype)
        q = ops.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        k = ops.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        v = ops.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        out = model(q, k, v)
        self.assertEqual(q.size(), out[0].size())
        self.assertEqual(dtype, out[0].dtype)

    def test_multihead_attention_dtype_batch_first(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        for training in (True, False):
            model = MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda().to(dtype)
            if not training:
                model = model.eval()
            q = ops.randn(bs, sl, embed_dim, device=device, dtype=dtype)
            k = ops.randn(bs, sl, embed_dim, device=device, dtype=dtype)
            v = ops.randn(bs, sl, embed_dim, device=device, dtype=dtype)
            # fast path currently doesn't support weights
            out = model(q, k, v, need_weights=False)
            self.assertEqual(q.size(), out[0].size())
            self.assertEqual(dtype, out[0].dtype)

    def test_multihead_attn_fast_path_query_and_bias_have_different_dtypes(self, device, dtype):
        mha = MultiheadAttention(4, 4, batch_first=True, dtype=dtype, device=device).eval()
        mha.in_proj_bias = Parameter(mha.in_proj_bias.to(torch.half).to(device))
        query = ops.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    def test_multihead_attn_fast_path_small_test(self, device, dtype):
        mha = MultiheadAttention(4, 4, batch_first=True, dtype=dtype, device=device).eval()
        query = ops.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    def test_multihead_attn_in_proj_bias_none(self, device, dtype):
        mha = MultiheadAttention(2, 2, bias=False, dtype=dtype, device=device)
        query = ops.rand(2, 2, 2, dtype=dtype, device=device)
        mha(query, query, query)

    def test_multihead_attn_in_proj_weight_none(self, device, dtype):
        # Setting kdim == vdim == 2 means that vdim != embed_dim
        # will cause the logic to use per-input project weights, thereby
        # forcing self.in_proj_weight = None
        mha = MultiheadAttention(4, 4, vdim=2, kdim=2, dtype=dtype, device=device)
        query = ops.rand(4, 4, 4, dtype=dtype, device=device)
        key = ops.rand(4, 4, 2, dtype=dtype, device=device)
        mha(query, key, key)

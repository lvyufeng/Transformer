import unittest
import numpy as np
import torch
import mindspore
from mindspore import Tensor
from ddt import ddt, data, unpack
from src.nn import MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer

dtype_list = [(mindspore.float32, torch.float32)]

if torch.cuda.is_available():
    dtype_list.append((mindspore.float16, torch.float16))

@ddt
class TestPytorchCompare(unittest.TestCase):
    @data(*dtype_list)
    @unpack
    def test_multi_head_attention_comp(self, ms_dtype, pt_dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        ms_model = MultiheadAttention(embed_dim, num_heads).to_float(ms_dtype)
        pt_model = torch.nn.MultiheadAttention(embed_dim, num_heads, dtype=pt_dtype)
        # data
        q = np.random.randn(sl, bs, embed_dim)
        k = np.random.randn(sl, bs, embed_dim)
        v = np.random.randn(sl, bs, embed_dim)
        # prepare data
        ms_q = Tensor(q, ms_dtype)
        ms_k = Tensor(k, ms_dtype)
        ms_v = Tensor(v, ms_dtype)
        pt_q = torch.tensor(q, dtype=pt_dtype)
        pt_k = torch.tensor(k, dtype=pt_dtype)
        pt_v = torch.tensor(v, dtype=pt_dtype)
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy(), mindspore.float32))

        ms_out = ms_model(ms_q, ms_k, ms_v)
        pt_out = pt_model(pt_q, pt_k, pt_v)
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-3, 1e-3)
        # print(ms_dtype, pt_dtype)

    @data(*dtype_list)
    @unpack
    def test_multi_head_attention_comp_qkv_equal(self, ms_dtype, pt_dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        ms_model = MultiheadAttention(embed_dim, num_heads).to_float(ms_dtype)
        pt_model = torch.nn.MultiheadAttention(embed_dim, num_heads, dtype=pt_dtype)
        # data
        q = np.random.randn(sl, bs, embed_dim)
        # prepare data
        ms_q = Tensor(q, ms_dtype)
        pt_q = torch.tensor(q, dtype=pt_dtype)
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy(), mindspore.float32))

        ms_out = ms_model(ms_q, ms_q, ms_q)
        pt_out = pt_model(pt_q, pt_q, pt_q)
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-3, 1e-3)
        # print(ms_dtype, pt_dtype)

    @data(*dtype_list)
    @unpack
    def test_transformer_encoder_layer(self, ms_dtype, pt_dtype):
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32
        nhead = 8

        ms_model = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to_float(ms_dtype)

        pt_model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True, dtype=pt_dtype)

        ms_model.set_train(False)
        pt_model.eval()

        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'gamma' in key:
                key = key.replace('gamma', 'weight')
            if 'beta' in key:
                key = key.replace('beta', 'bias')
            param.set_data(Tensor(pt_params.get(key).detach().numpy(), mindspore.float32))

        src = np.random.rand(batch_size, seqlen, d_model)
        src_mask = np.zeros((seqlen, seqlen))

        src_ms = Tensor(src, ms_dtype)
        src_mask_ms = Tensor(src_mask, mindspore.bool_)
        src_pt = torch.tensor(src, dtype=pt_dtype)
        src_mask_pt = torch.tensor(src_mask, dtype=torch.bool)

        ms_out = ms_model(src_ms, src_mask=src_mask_ms)
        pt_out = pt_model(src_pt, src_mask=src_mask_pt)
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-3, 1e-3)
        # print(ms_dtype, pt_dtype)
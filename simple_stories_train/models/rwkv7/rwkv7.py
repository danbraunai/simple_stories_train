import os
import sys
import uuid
import glob
import time, datetime, wandb
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import cached_file

import argparse, random, math

from torch.utils.cpp_extension import load


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b,chunk_len):
        B,T,H,C = w.shape
        assert T%chunk_len == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        w,q,k,v,z,b = [i.contiguous() for i in [w,q,k,v,z,b]]
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//chunk_len,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert dy.dtype == torch.bfloat16
        dy = dy.contiguous()
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b,chunk_len):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b,chunk_len).view(B,T,HC)


class RWKV7_Tmix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,args.dim_att))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_MIX_LORA = 28
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(ortho_init(torch.zeros(4, D_MIX_LORA, args.n_embd), 0.1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, args.dim_att), 0.1))

            D_AAA_LORA = 16
            self.time_aaa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, args.dim_att), 0.1))

            D_KKK_LORA = 16
            self.time_kkk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(ortho_init(torch.zeros(D_KKK_LORA, args.dim_att), 0.1))

            D_GATE_LORA = 120
            self.gate_w1 = nn.Parameter(torch.zeros(args.n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, args.dim_att), 0.1))

            D_MA_LORA = 16
            self.ma_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, args.dim_att), 0.1))
            self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
            D_MK_LORA = 16
            self.mk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, args.dim_att), 0.1))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))
            if layer_id != 0:
                D_MV_LORA = 16
                self.mv_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MV_LORA))
                self.mv_w2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, args.dim_att), 0.1))
                self.time_misc_v = nn.Parameter(torch.zeros(1,1,args.n_embd)+1.0)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.key.weight.data.uniform_(-0.05/(self.n_embd**0.5), 0.05/(self.n_embd**0.5))
            self.value.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.output.weight.data.zero_()

    def forward(self, x, v1):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        xrg, xwa, xk, xv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + xrg)
        xwa = x + xx * (self.time_maa_wa + xwa)
        xk = x + xx * (self.time_maa_k + xk)
        xv = x + xx * (self.time_maa_v + xv)

        r = self.receptance(xrg)
        w = -F.softplus(-(self.time_decay + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v1 = v
        else:
            v = v + (v1 - v) * torch.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        g = torch.sigmoid(xrg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        a = torch.sigmoid(self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2)

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k*a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * torch.clamp(w*mk, max=0).exp()

        x = RUN_CUDA_RWKV7g(r.bfloat16(), w.bfloat16(), k.bfloat16(), v.bfloat16(), -kk.bfloat16(), (kk*a).bfloat16(), self.args.chunk_len)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v1

class RWKV7_CMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 7 * config.n_embd // 2, bias=False)
        self.c_proj = nn.Linear(7 * config.n_embd // 2, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.attn = RWKV7_Tmix(config, layer_id)
        self.mlp = RWKV7_CMix(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(self.ln1(x), v1)
        x = x + x1
        x = x + self.mlp(self.ln2(x))
        return x, v1


class RWKV7Config(PretrainedConfig):
    model_type = "rwkv7"

    def init(self):
        #vocab_size : int = 50304
        self.vocab_size : int = 3072
        self.n_layer : int = 12
        self.n_head : int = 6
        self.n_embd : int = 768
        self.head_size : int = 64
        self.chunk_len : int = 33
    
    @classmethod
    def register_for_auto_class(cls):
        from transformers import AutoConfig, AutoModel
        AutoConfig.register(cls.model_type, cls)
        AutoModel.register(cls.model_type, cls)


class RWKV7(PreTrainedModel):
    config_class = RWKV7Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        CHUNK_LEN = 1
        head_size = 64
        config = RWKV7Config(vocab_size=3072, n_layer=16, n_head=768 // head_size, n_embd=768, head_size=head_size)

        CUDA_FLAGS = ["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
        
        sources = [
            cached_file(model_dir, f'rwkv_cuda_wind/backstepping_f32_{1 if HEAD_SIZE < 128 else 2}.cu', token=kwargs['token']),
            cached_file(model_dir, 'rwkv_cuda_wind/backstepping_f32.cpp', token=kwargs['token']),
        ]
        load(name="wind_backstepping", sources=sources, is_python_module=False, verbose=True, extra_cuda_cflags=CUDA_FLAGS+[f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}"])

        #print(config)
        model = cls(config)
        #weights_path = f"{model_dir}/pytorch_model.bin"
        weights_path = cached_file(model_dir, 'pytorch_model.bin', token=kwargs['token'])
        state_dict = torch.load(weights_path, weights_only=True)['model']
        for key in list(state_dict.keys()):
            state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        model = model.cuda().bfloat16()
        return model

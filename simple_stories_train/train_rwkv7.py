import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
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

from dataloaders import DatasetConfig, create_data_loader
from models.rwkv7.rwkv7 import *
from pydantic import Field

import argparse, random, math


@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = '../../simplestories/simplestories_train_*.bin' # input .bin to train on
    input_val_bin : str = '../../simplestories/simplestories_val_*.bin' # input .bin to eval validation loss on

    # optimization hyperparams
    batch_size : int = 512 # batch size, in sequences, across all devices
    device_batch_size : int = 32 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 3200 * 10 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 914 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0

    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 125 # every how many steps to save the checkpoint? 0 for only at the end

    # learning rates
    muon_lr : float = 0.02
    adam_lr : float = 0.0026
    ln_lr : float = 0.0090


def get_optimizers(raw_model, args):
    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.3, betas=(0.9, 0.95), fused=True)
    optimizer1.my_name = 'Adam-wte'

    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.002, betas=(0.9, 0.95), fused=True)
    optimizer2.my_name = 'Adam-head'

    params = list(raw_model.transformer.h.named_parameters())
    optimizer3 = Muon([p for n,p in params if p.ndim == 2 and '_w1' not in n and '_w2' not in n], lr=args.muon_lr, momentum=0.95)
    optimizer3.my_name = 'Muon !!!'

    optimizer4 = torch.optim.Adam([p for n,p in params if (p.ndim != 2 or '_w1' in n or '_w2' in n) and ('lambdas' not in n and 'ln' not in n)], lr=args.adam_lr, betas=(0.9, 0.95), fused=True)
    optimizer4.my_name = 'Adam'

    optimizer5 = torch.optim.Adam([p for n,p in params if 'lambdas' in n], lr=0.02, betas=(0.9, 0.95), fused=True)
    optimizer5.my_name = 'Adam-s'

    optimizer6 = torch.optim.Adam([p for n,p in params if 'ln' in n], lr=args.ln_lr, betas=(0.9, 0.95), fused=True)
    optimizer6.my_name = 'Adam-LN'

    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr_func(args.num_iterations, args.warmup_iters, args.warmdown_iters)) for opt in optimizers]
    return optimizers, schedulers


# learning rate decay scheduler (linear warmup and warmdown)
def get_lr_func(num_iter, warmup_iters, warmdown_iters):
    def get_lr(it):
        assert it <= num_iter
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return (it+1) / warmup_iters
        # 2) constant lr for a while
        elif it < num_iter - warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (num_iter - it) / warmdown_iters
            return decay_ratio
    return get_lr


# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        self.zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = self.zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headsz', type=int, default=64) # increase to 96/128/192 for better loss (slow in inefficient implementation, fast after optimization)
    parser.add_argument('--muon_lr', type=float, default=0.02)
    parser.add_argument('--adam_lr', type=float, default=0.0026) # adam lr for misc weights (lora, time, etc.)
    parser.add_argument('--ln_lr', type=float, default=0.0090)
    parser.add_argument('--device_bsz', type=int, default=64)
    parser.add_argument('--bsz', type=int, default=8*64)
    parser.add_argument('--fast_cuda', action=argparse.BooleanOptionalAction) # much faster cuda
    parser.add_argument('--wind_cuda', action=argparse.BooleanOptionalAction) # even faster cuda, likely worse loss
    parser.add_argument('--random_seed', type=int, default=-1)
    cmd_args = parser.parse_args()

    if cmd_args.random_seed != -1:
        random.seed(cmd_args.random_seed)
        np.random.seed(cmd_args.random_seed)
        torch.manual_seed(cmd_args.random_seed)
        torch.cuda.manual_seed_all(cmd_args.random_seed)

    args = Hyperparameters()

    args.headsz = cmd_args.headsz
    args.muon_lr = cmd_args.muon_lr
    args.adam_lr = cmd_args.adam_lr
    args.ln_lr = cmd_args.ln_lr

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    val_steps = 20
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    train_dataset_config: DatasetConfig = DatasetConfig(
        name="lennart-finke/SimpleStories",
        is_tokenized=False,
        tokenizer_file_path="tokenizer/stories-3072.json",
        streaming=True,
        split="train",
        n_ctx=1024,
        seed=0,
        column_name="story",
    )
    val_dataset_config: DatasetConfig = DatasetConfig(
        name="lennart-finke/SimpleStories",
        is_tokenized=False,
        tokenizer_file_path="tokenizer/stories-3072.json",
        streaming=True,
        split="test",
        n_ctx=1024,
        seed=0,
        column_name="story",
    )

    train_loader, _ = create_data_loader(
        dataset_config=train_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    train_loader = iter(train_loader)  # Is this the right way to sample from a Pytorch DataLoader?
    bat = next(train_loader)["input_ids"].to(torch.int64)
    x = bat.view(B, T)[:, :-1]  # inputs
    y = bat.view(B, T)[:, 1:]  # targets

    val_loader, _ = create_data_loader(
        dataset_config=val_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    val_loader = iter(val_loader)  # Is this the right way to sample from a Pytorch DataLoader?

    num_vocab = 3072
    config = RWKV7Config(vocab_size=num_vocab, n_layer=16, n_head=768 // args.headsz, n_embd=768, head_size=args.headsz, chunk_len=33)

    CUDA_FLAGS = ["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    
    sources = [
        f'models/rwkv7/rwkv_cuda_wind/backstepping_f32_{1 if head_size < 128 else 2}.cu',
        'models/rwkv7/rwkv_cuda_wind/backstepping_f32.cpp'
    ]
    load(name="wind_backstepping", sources=sources, is_python_module=False, verbose=True, extra_cuda_cflags=CUDA_FLAGS+[f'-D_C_={head_size}', f"-D_CHUNK_LEN_={config.chunk_len}"])

    model = RWKV7(config)
    model = model.cuda()
    torch._dynamo.config.optimize_ddp = False # otherwise compiler will complain
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    # config.max_autotune = True # faster, but VERY slow to compile
    model = torch.compile(model, fullgraph=True)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(True)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    optimizers, schedulers = get_optimizers(raw_model, args)

    if master_process:
        n_params = 0
        n_found = []
        n_all = []
        for n,p in raw_model.named_parameters():
            n_all.append(n)
            n_params += p.numel()
            found = False
            for o in optimizers:
                for group in o.param_groups:
                    for pp in group['params']:
                        if p.data_ptr() == pp.data_ptr():
                            n_found.append(n)
                            found = True
                            print(o.my_name.ljust(10), str(list(p.shape)).ljust(20), n)
            if not found:
                print('MISSING optimizer:', n)
                exit(1)
        print(n_all)
        print(n_all)
        print(list(set(n_all) - set(n_all)))
        print('model params', n_params)
    # begin logging
    if master_process:
        run_id = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        run_prefix = 'v7wind' if cmd_args.wind_cuda else ('v7fast' if cmd_args.fast_cuda else 'v7')
        if cmd_args.random_seed != -1:
            run_prefix += f' seed{cmd_args.random_seed}'
        wandb.init(
            project='fast-nanogpt',
            name=f'{run_prefix} {args.adam_lr}/{args.muon_lr}/{args.ln_lr} {run_id}',
            config=args,
            save_code=False,
        )
        logdir = 'logs/%s/' % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = 'logs/%s.txt' % run_id

        # create the log file
        with open(logfile, "w") as f:
            f.write(str(cmd_args) + '\n')

            # begin the log by printing this file (the Python code)
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')

            # log information about the hardware/software environment this is running on
            # and print the full `nvidia-smi` to file
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')

    training_time_ms = 0

    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    tokens_processed = 0

    # begin training
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader_iter = iter(
                val_loader
            )  # By creating the iterator anew, we sample the same data each time
            val_loss = 0.0
            for _ in range(val_steps):
                bat = next(val_loader_iter)["input_ids"].to(torch.int64)
                x_val = bat.view(B, T)[:, :-1] # inputs
                y_val = bat.view(B, T)[:, 1:] # targets
                with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
                wandb.log({"loss_val": val_loss.item()}, step=int(step+1))
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps+1):
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()

            # advance the dataset for the next batch
            bat = next(train_loader)["input_ids"].to(torch.int64)
            x = bat.view(B, T)[:, :-1]  # inputs
            y = bat.view(B, T)[:, 1:]  # targets
            tokens_processed += x.shape[0] * x.shape[1]

            # backward pass
            if i < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward() # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps

        # momentum warmup for Muon
        frac = min(step/500, 1)
        optimizers[2].param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        # step the optimizers and schedulers
        lr = []
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
            lr.append(sched.get_last_lr())

        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms total tokens:{tokens_processed}")
            with open(logfile, "a") as f:
                f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms total tokens:{tokens_processed}\n")
            wandb.log({"loss": train_loss.item(), "lr1": float(lr[0][0]), "lr2": float(lr[1][0]), "step_t": approx_time/timed_steps, "tokens": tokens_processed}, step=int(step+1))

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
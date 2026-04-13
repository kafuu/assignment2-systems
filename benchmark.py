import argparse
import timeit
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
import time

import math
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import rearrange, einsum
import torch.nn.functional as F


from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import clip_gradient
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_systems.flash_attention import FlashAttentionTriton

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]

    #计算分数
    with nvtx.range("Attention: computing scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("Attention: computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("Attention: computing out"):
        out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return out

def triton_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    prefix = Q.shape[:-2]
    Nq, D = Q.shape[-2:]
    Nk = K.shape[-2]
    Dv = V.shape[-1]

    batch = math.prod(prefix) if len(prefix) > 0 else 1

    Q_ = Q.reshape(batch, Nq, D)
    K_ = K.reshape(batch, Nk, D)
    V_ = V.reshape(batch, Nk, Dv)

    # 这里只是为了先跑通 benchmark
    is_causal = mask is not None

    with torch.no_grad():
        O_ = FlashAttentionTriton.apply(Q_, K_, V_, is_causal)

    return O_.reshape(*prefix, Nq, Dv)


def py_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    print(">>> py_attention called")
    is_causal = mask is not None

    # 如果你的 mask 只是 causal mask，那么直接用 is_causal 即可
    # 不把 mask 真传进去，避免和 is_causal 冲突
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    

cs336_basics.model.scaled_dot_product_attention = py_attention

@nvtx.range("bench mark")
def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------------------------------------------
    # 1. 实例化模型 (根据传入的参数)
    # -----------------------------------------------------------------
    print(f"Initializing model with d_model={args.d_model}, layers={args.num_layers}...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    )
    model.to(device)
    model.train() # 基准测试通常在 train 模式下进行，确保 Dropout 等正常运作

    if args.auto_compile:
        model.compile()
    
    # -----------------------------------------------------------------
    # 2. 生成随机假数据 (Dummy Data)
    # -----------------------------------------------------------------
    # 生成输入 x: 形状 (batch_size, context_length)，范围 [0, vocab_size)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    
    # 如果需要测试反向传播，通常需要一个随机的目标值 y 来算 Loss
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    # -----------------------------------------------------------------
    # 3. 预热阶段 (Warm-up) - 建立缓存，不计入时间
    # -----------------------------------------------------------------
    print(f"Running {args.warmup_steps} warm-up steps...")
    context = nullcontext() if args.measure_backward else torch.no_grad()
    for _ in range(args.warmup_steps):
        with context:
            logits = model(x)
        if args.measure_backward:
            loss = cross_entropy(logits, y) 
            loss.backward()
        
        # 强制同步，确保每一步的计算真实完成
        torch.cuda.synchronize()

    # -----------------------------------------------------------------
    # 4. 测量阶段 (Measurement)
    # -----------------------------------------------------------------
    print(f"Measuring {args.num_steps} steps...")
    timings = []

    if args.measure_backward:
        optimizer = AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    for _ in range(args.num_steps):
        # 记录起点
        start_time = timeit.default_timer()
        
        # 执行前向 (Forward)
        with nvtx.range("Benchmark: forwards"):
            with context:
                logits = model(x)
            torch.cuda.synchronize()    
        

        if args.measure_backward:
            optimizer.zero_grad()   
            # 执行反向 (Backward)
            with nvtx.range("Benchmark: backwards"):
                    loss = cross_entropy(logits, y)  # 极简版的梯度回传触发器
                    loss.backward()
                    torch.cuda.synchronize()

            # 梯度裁剪
            clip_gradient(model.parameters(), 1.0)
            
            with nvtx.range("Benchmark: step"):
                optimizer.step()
                torch.cuda.synchronize()
            
        # 记录终点 (必须先同步 GPU)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        
        step_time = end_time - start_time
        timings.append(step_time)

    # -----------------------------------------------------------------
    # 5. 统计与报告
    # -----------------------------------------------------------------
    print("peak allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")
    print("peak reserved :", torch.cuda.max_memory_reserved() / 1024**3, "GB")
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    
    print(f"--- Benchmark Results ---")
    print(f"Mode: {'Forward + Backward' if args.measure_backward else 'Forward Only'}")
    print(f"Average Time per step: {avg_time:.4f} seconds")
    print(f"Standard Deviation:    {std_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Benchmarking Script")
    
    # 模型架构参数 (对应 Table 1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--rope_theta", type=int, default=10000)
    
    # 全局常量参数 (作业规定)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    
    # 测量逻辑参数
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--measure_backward", action="store_true", help="Include backward pass in measurement")
    
    parser.add_argument("--record_mem", action="store_true")
    parser.add_argument("--auto_precision", action="store_true")
    parser.add_argument("--auto_compile", action="store_true")

    
    args = parser.parse_args()
    context = torch.autocast(device_type="cuda",dtype=torch.bfloat16) if args.auto_precision else nullcontext()
    with context :
        if args.record_mem:
            # 记录显存使用
            print("recording MEM")
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        
        run_benchmark(args)

        if args.record_mem:
            #保存为文件
            print("saving MEM")
            torch.cuda.memory._dump_snapshot(f"MEM_snapshot_ctx{args.context_length}_d{args.d_model}_atprec{args.auto_precision}_back{args.measure_backward}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
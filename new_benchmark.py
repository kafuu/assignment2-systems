import argparse
import csv
import math
import timeit
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F
import triton.testing

from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import clip_gradient
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_systems.flash_attention import FlashAttentionTriton


# ============================================================
# 1. Attention implementations for model-level benchmark
# ============================================================

@nvtx.range("regular scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Regular PyTorch attention.
    This explicitly materializes attention_scores and attention_weights.
    """
    d_k = K.shape[-1]

    with nvtx.range("Attention: computing scores"):
        attention_scores = einsum(
            Q,
            K,
            "... query d_k, ... key d_k -> ... query key",
        ) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(
                mask,
                attention_scores,
                torch.full_like(attention_scores, float("-inf")),
            )

    with nvtx.range("Attention: computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("Attention: computing out"):
        out = einsum(
            attention_weights,
            V,
            "... query key, ... key d_v ->  ... query d_v",
        )

    return out


def triton_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    """
    Adapter used when replacing attention inside BasicsTransformerLM.

    Your FlashAttentionTriton expects [B, N, D].
    The model may pass tensors with extra leading dims, e.g. [B, H, N, D],
    so we flatten all leading dims into one batch dimension.
    """
    prefix = Q.shape[:-2]
    Nq, D = Q.shape[-2:]
    Nk = K.shape[-2]
    Dv = V.shape[-1]

    if Dv != D:
        raise ValueError(
            f"Current FlashAttentionTriton assumes Dv == D, got D={D}, Dv={Dv}"
        )

    batch = math.prod(prefix) if len(prefix) > 0 else 1

    Q_ = Q.reshape(batch, Nq, D).contiguous()
    K_ = K.reshape(batch, Nk, D).contiguous()
    V_ = V.reshape(batch, Nk, Dv).contiguous()

    is_causal = mask is not None

    # Important:
    # Do NOT wrap this in torch.no_grad().
    # Forward-only benchmark already uses no_grad outside.
    # Backward benchmark needs the graph.
    O_ = FlashAttentionTriton.apply(Q_, K_, V_, is_causal)

    return O_.reshape(*prefix, Nq, Dv)


def py_sdpa_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    """
    PyTorch official optimized SDPA.
    Do not call this "regular PyTorch attention" in the paper.
    """
    is_causal = mask is not None

    return F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )


def patch_model_attention(attention_impl: str):
    """
    Patch cs336_basics.model.scaled_dot_product_attention for model benchmark.
    """
    if attention_impl == "torch_regular":
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    elif attention_impl == "torch_sdpa":
        cs336_basics.model.scaled_dot_product_attention = py_sdpa_attention
    elif attention_impl == "triton":
        cs336_basics.model.scaled_dot_product_attention = triton_attention
    else:
        raise ValueError(f"Unknown model attention_impl: {attention_impl}")


# ============================================================
# 2. Attention-only implementations for operator-level benchmark
# ============================================================

def attention_torch_regular(Q, K, V, is_causal=True):
    """
    Regular PyTorch baseline required by the assignment.

    It explicitly does:
        S = Q K^T / sqrt(D)
        P = softmax(S)
        O = P V

    Do NOT use F.scaled_dot_product_attention here, because that may dispatch
    to FlashAttention or memory-efficient attention internally.
    """
    D = Q.shape[-1]
    Nq = Q.shape[-2]
    Nk = K.shape[-2]

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    if is_causal:
        q_idx = torch.arange(Nq, device=Q.device)[:, None]
        k_idx = torch.arange(Nk, device=Q.device)[None, :]
        mask = k_idx <= q_idx
        scores = torch.where(
            mask[None, :, :],
            scores,
            torch.full_like(scores, float("-inf")),
        )

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, V)
    return out


def attention_torch_sdpa(Q, K, V, is_causal=True):
    """
    PyTorch official optimized SDPA. Optional extra comparison.
    """
    return F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )


def attention_triton_flash(Q, K, V, is_causal=True):
    """
    Your Triton FlashAttention implementation.
    Q/K/V shape: [B, N, D]
    """
    return FlashAttentionTriton.apply(Q, K, V, is_causal)


def get_attention_impl(name: str):
    if name == "torch_regular":
        return attention_torch_regular
    if name == "torch_sdpa":
        return attention_torch_sdpa
    if name == "triton":
        return attention_triton_flash
    raise ValueError(f"Unknown attention_impl: {name}")


# ============================================================
# 3. Attention-only benchmark helpers
# ============================================================

def make_qkv(batch_size, seq_len, head_dim, dtype, device):
    Q = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    V = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dO = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )
    return Q, K, V, dO


def clear_cuda():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_attention_forward(attn_fn, Q, K, V, warmup_steps, num_steps):
    """
    Measure attention forward latency only:
        O = attention(Q, K, V)
    """
    Q_ = Q.detach()
    K_ = K.detach()
    V_ = V.detach()

    def run():
        with torch.no_grad():
            attn_fn(Q_, K_, V_, True)

    return triton.testing.do_bench(
        run,
        warmup=warmup_steps,
        rep=num_steps,
    )


def bench_attention_backward(attn_fn, Q, K, V, dO, warmup_steps, num_steps):
    """
    Measure backward latency only.

    Build the forward graph once, then do_bench only measures:
        O.backward(dO, retain_graph=True)
    """
    O = attn_fn(Q, K, V, True)

    def run():
        Q.grad = None
        K.grad = None
        V.grad = None
        O.backward(dO, retain_graph=True)

    return triton.testing.do_bench(
        run,
        warmup=warmup_steps,
        rep=num_steps,
    )


def bench_attention_fwd_bwd(attn_fn, Q, K, V, dO, warmup_steps, num_steps):
    """
    Measure end-to-end forward + backward:
        O = attention(Q, K, V)
        O.backward(dO)
    """
    def run():
        Q.grad = None
        K.grad = None
        V.grad = None
        O = attn_fn(Q, K, V, True)
        O.backward(dO)

    return triton.testing.do_bench(
        run,
        warmup=warmup_steps,
        rep=num_steps,
    )


def save_attention_result(result, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "impl",
        "seq_len",
        "head_dim",
        "dtype",
        "forward_ms",
        "backward_ms",
        "fwd_bwd_ms",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "status",
    ]

    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def run_attention_benchmark(args):
    """
    Operator-level benchmark required by the assignment.

    It benchmarks only:
        Q, K, V -> O

    It does NOT benchmark the whole Transformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Attention benchmark requires CUDA."

    attn_fn = get_attention_impl(args.attention_impl)

    if args.attn_dtype == "bf16":
        dtype = torch.bfloat16
    elif args.attn_dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {args.attn_dtype}")

    batch_size = 1
    seq_len = args.context_length
    head_dim = args.head_dim

    print("==================================================")
    print("Attention-only Benchmark")
    print(f"impl       : {args.attention_impl}")
    print(f"seq_len    : {seq_len}")
    print(f"head_dim   : {head_dim}")
    print(f"dtype      : {dtype}")
    print(f"batch_size : {batch_size}")
    print(f"causal     : True")
    print("==================================================")

    result = {
        "impl": args.attention_impl,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": args.attn_dtype,
        "forward_ms": "",
        "backward_ms": "",
        "fwd_bwd_ms": "",
        "peak_allocated_gb": "",
        "peak_reserved_gb": "",
        "status": "ok",
    }

    try:
        peak_allocated = 0.0
        peak_reserved = 0.0

        # -------------------------
        # Forward
        # -------------------------
        clear_cuda()
        Q, K, V, dO = make_qkv(batch_size, seq_len, head_dim, dtype, device)

        forward_ms = bench_attention_forward(
            attn_fn,
            Q,
            K,
            V,
            args.warmup_steps,
            args.num_steps,
        )

        torch.cuda.synchronize()
        peak_allocated = max(
            peak_allocated,
            torch.cuda.max_memory_allocated() / 1024**3,
        )
        peak_reserved = max(
            peak_reserved,
            torch.cuda.max_memory_reserved() / 1024**3,
        )

        # -------------------------
        # Backward only
        # -------------------------
        clear_cuda()
        Q, K, V, dO = make_qkv(batch_size, seq_len, head_dim, dtype, device)

        backward_ms = bench_attention_backward(
            attn_fn,
            Q,
            K,
            V,
            dO,
            args.warmup_steps,
            args.num_steps,
        )

        torch.cuda.synchronize()
        peak_allocated = max(
            peak_allocated,
            torch.cuda.max_memory_allocated() / 1024**3,
        )
        peak_reserved = max(
            peak_reserved,
            torch.cuda.max_memory_reserved() / 1024**3,
        )

        # -------------------------
        # Forward + Backward
        # -------------------------
        clear_cuda()
        Q, K, V, dO = make_qkv(batch_size, seq_len, head_dim, dtype, device)

        fwd_bwd_ms = bench_attention_fwd_bwd(
            attn_fn,
            Q,
            K,
            V,
            dO,
            args.warmup_steps,
            args.num_steps,
        )

        torch.cuda.synchronize()
        peak_allocated = max(
            peak_allocated,
            torch.cuda.max_memory_allocated() / 1024**3,
        )
        peak_reserved = max(
            peak_reserved,
            torch.cuda.max_memory_reserved() / 1024**3,
        )

        result["forward_ms"] = f"{forward_ms:.4f}"
        result["backward_ms"] = f"{backward_ms:.4f}"
        result["fwd_bwd_ms"] = f"{fwd_bwd_ms:.4f}"
        result["peak_allocated_gb"] = f"{peak_allocated:.4f}"
        result["peak_reserved_gb"] = f"{peak_reserved:.4f}"

        print("--- Attention Benchmark Results ---")
        print(f"Forward latency:        {forward_ms:.4f} ms")
        print(f"Backward latency:       {backward_ms:.4f} ms")
        print(f"Forward+Backward:       {fwd_bwd_ms:.4f} ms")
        print(f"Peak allocated memory:  {peak_allocated:.4f} GB")
        print(f"Peak reserved memory:   {peak_reserved:.4f} GB")

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        torch.cuda.empty_cache()
        print("OOM")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "OOM"
            torch.cuda.empty_cache()
            print("OOM")
        else:
            result["status"] = f"ERR: {repr(e)}"
            print(result["status"])

    if args.output_csv is not None:
        save_attention_result(result, args.output_csv)

    return result


# ============================================================
# 4. Model-level benchmark: your original benchmark
# ============================================================

@nvtx.range("bench mark")
def run_model_benchmark(args):
    """
    Full Transformer benchmark.

    This is your original benchmark:
        input_ids -> model -> logits

    This is NOT the assignment's operator-level benchmark, but it is useful
    for your thesis as an end-to-end validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_model_attention(args.attention_impl)

    print(f"Model-level attention impl: {args.attention_impl}")

    print(f"Initializing model with d_model={args.d_model}, layers={args.num_layers}...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    model.to(device)
    model.train()

    if args.auto_compile:
        model.compile()

    x = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    y = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    print(f"Running {args.warmup_steps} warm-up steps...")

    context = nullcontext() if args.measure_backward else torch.no_grad()

    if args.measure_backward:
        optimizer = AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    else:
        optimizer = None

    for _ in range(args.warmup_steps):
        if args.measure_backward:
            optimizer.zero_grad()

        with context:
            logits = model(x)

        if args.measure_backward:
            loss = cross_entropy(logits, y)
            loss.backward()
            clip_gradient(model.parameters(), 1.0)
            optimizer.step()

        torch.cuda.synchronize()

    print(f"Measuring {args.num_steps} steps...")
    timings = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(args.num_steps):
        start_time = timeit.default_timer()

        if args.measure_backward:
            optimizer.zero_grad()

        with nvtx.range("Benchmark: forwards"):
            with context:
                logits = model(x)
            torch.cuda.synchronize()

        if args.measure_backward:
            with nvtx.range("Benchmark: backwards"):
                loss = cross_entropy(logits, y)
                loss.backward()
                torch.cuda.synchronize()

            clip_gradient(model.parameters(), 1.0)

            with nvtx.range("Benchmark: step"):
                optimizer.step()
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        timings.append(end_time - start_time)

    print("peak allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")
    print("peak reserved :", torch.cuda.max_memory_reserved() / 1024**3, "GB")

    avg_time = np.mean(timings)
    std_time = np.std(timings)

    print("--- Model Benchmark Results ---")
    print(f"Mode: {'Forward + Backward + Optimizer Step' if args.measure_backward else 'Forward Only'}")
    print(f"Average Time per step: {avg_time:.4f} seconds")
    print(f"Standard Deviation:    {std_time:.4f} seconds")


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer / Attention Benchmarking Script")

    # ------------------------------
    # Benchmark target
    # ------------------------------
    parser.add_argument(
        "--benchmark_target",
        type=str,
        default="model",
        choices=["model", "attention"],
        help="model: benchmark full Transformer; attention: benchmark attention operator only",
    )

    # ------------------------------
    # Model architecture params
    # ------------------------------
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--rope_theta", type=int, default=10000)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)

    # ------------------------------
    # Benchmark params
    # ------------------------------
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)

    parser.add_argument(
        "--measure_backward",
        action="store_true",
        help="For model benchmark, include backward pass and optimizer step",
    )

    parser.add_argument("--record_mem", action="store_true")
    parser.add_argument("--auto_precision", action="store_true")
    parser.add_argument("--auto_compile", action="store_true")

    # ------------------------------
    # Attention implementation
    # Used by both model benchmark and attention-only benchmark.
    # ------------------------------
    parser.add_argument(
        "--attention_impl",
        type=str,
        default="triton",
        choices=["torch_regular", "torch_sdpa", "triton"],
        help=(
            "Attention implementation. "
            "For assignment baseline use torch_regular vs triton. "
            "torch_sdpa is optional official optimized comparison."
        ),
    )

    # ------------------------------
    # Attention-only params
    # ------------------------------
    parser.add_argument(
        "--head_dim",
        type=int,
        default=64,
        help="Head dimension for attention-only benchmark",
    )

    parser.add_argument(
        "--attn_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="dtype for attention-only benchmark",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Append attention benchmark result to CSV",
    )

    args = parser.parse_args()

    if args.benchmark_target == "attention":
        run_attention_benchmark(args)

    else:
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.auto_precision
            else nullcontext()
        )

        with context:
            if args.record_mem:
                print("recording MEM")
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            run_model_benchmark(args)

            if args.record_mem:
                print("saving MEM")
                torch.cuda.memory._dump_snapshot(
                    f"MEM_snapshot_ctx{args.context_length}_d{args.d_model}_"
                    f"atprec{args.auto_precision}_back{args.measure_backward}.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)
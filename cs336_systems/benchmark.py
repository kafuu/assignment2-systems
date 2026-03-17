import argparse
import timeit
import torch
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

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
    model.train() # 基准测试通常在 train 模式下进行，确保 Dropout 等正常运作
    
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
    for _ in range(args.warmup_steps):
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
    
    for _ in range(args.num_steps):
        # 记录起点
        start_time = timeit.default_timer()
        
        # 执行前向 (Forward)
        logits = model(x)
        
        # 执行反向 (Backward)
        if args.measure_backward:
            loss = logits.sum() # 极简版的梯度回传触发器
            loss.backward()
            pass
            
        # 记录终点 (必须先同步 GPU)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        
        step_time = end_time - start_time
        timings.append(step_time)

    # -----------------------------------------------------------------
    # 5. 统计与报告
    # -----------------------------------------------------------------
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
    
    args = parser.parse_args()
    run_benchmark(args)
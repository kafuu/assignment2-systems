#!/bin/bash

OUT="results/attention_benchmark.csv"
rm -f ${OUT}

IMPLS=("torch_regular" "triton")
SEQ_LENS=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
HEAD_DIMS=(16 32 64 128)
DTYPES=("bf16" "fp32")

for dtype in "${DTYPES[@]}"; do
  for seq in "${SEQ_LENS[@]}"; do
    for hd in "${HEAD_DIMS[@]}"; do
      for impl in "${IMPLS[@]}"; do
        echo "=================================================="
        echo "impl=${impl}, seq=${seq}, head_dim=${hd}, dtype=${dtype}"
        echo "=================================================="

        python new_benchmark.py \
          --benchmark_target attention \
          --attention_impl ${impl} \
          --context_length ${seq} \
          --head_dim ${hd} \
          --attn_dtype ${dtype} \
          --warmup_steps 5 \
          --num_steps 20 \
          --output_csv ${OUT}
      done
    done
  done
done
#!/bin/bash

#1. 256下前向和全流程
#2. 前向下128,256,512
#3. 256下前向和全流程，不使用BF16，与使用对比

#测试类型
MODE="test"
# 定义你要测试的上下文长度数组
CONTEXT_LENGTHS=(4096 8192 16384 32768 65536 131072)
# 定义你要测试的模型尺寸
MODEL_SIZES=("small")
# 模型自动精度
AUTO_PRECISION="false"
# 显存记录
MEM_RECORD="false"
# 反向传播
BACK="false"
# 自动编译
AUTO_COMPILER="false"



MEM_FLAG=""
PRECISION_FLAG=""
BACK_FLAG=""
COMPILE_FLAG=""

if [ "$MEM_RECORD" = "true" ]; then
    MEM_FLAG="--record_mem"
fi

if [ "$AUTO_PRECISION" = "true" ]; then
    PRECISION_FLAG="--auto_precision"
fi

if [ "$BACK" = "true" ]; then
    BACK_FLAG="--measure_backward"
fi

if [ "$AUTO_COMPILER" = "true" ]; then
    COMPILE_FLAG="--auto_compile"
fi


for ctx in "${CONTEXT_LENGTHS[@]}"; do
    for size in "${MODEL_SIZES[@]}"; do
        
        # 根据模型尺寸，自动查表匹配对应的超参数
        case $size in
            "small")  d_m=768;  d_ff=3072;  nl=12; nh=12 ;;
            "medium") d_m=1024; d_ff=4096;  nl=24; nh=16 ;;
            "large")  d_m=1280; d_ff=5120;  nl=36; nh=20 ;;
            "xl")     d_m=1600; d_ff=6400;  nl=48; nh=25 ;;
            "2.7B")   d_m=2560; d_ff=10240; nl=32; nh=32 ;;
            *) echo "Unknown size"; exit 1 ;;
        esac

        case $MODE in
            "train") file_name="benchmark_train.py" ;;
            "test") file_name="benchmark.py" ;;
            *) echo "Unknown mode"; exit 1 ;;
        esac

        # 动态生成极其清晰的报告文件名，例如: result_small_ctx512
        OUT_FILE="result_${size}_ctx${ctx}_mode${MODE}_autoprec.${AUTO_PRECISION}_recmem.${MEM_RECORD}"
        
        echo "=================================================="
        echo "模式：${MODE}"
        echo "正在启动测试: 模型=${size}, Context=${ctx}", 自动精度=${AUTO_PRECISION}， 自动编译=${AUTO_COMPILER}, 反向传播=${BACK}
        echo "=================================================="

        # 执行你的 nsys 命令，将变量打入参数中
        /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
            -o ./out/${OUT_FILE} --force-overwrite true \
            python ./${file_name} \
            --d_model $d_m \
            --d_ff $d_ff \
            --num_layers $nl \
            --num_heads $nh \
            --vocab_size 10000 \
            --batch_size 4 \
            --context_length $ctx \
            --rope_theta 10000 \
            --warmup_steps 5 \
            --num_steps 20 \
            ${BACK_FLAG} \
            ${PRECISION_FLAG} \
            ${MEM_FLAG} \
            ${COMPILE_FLAG}
            
        # 可选：如果遇到 OOM，nsys 会报错，但脚本会继续跑下一个组合
    done
done

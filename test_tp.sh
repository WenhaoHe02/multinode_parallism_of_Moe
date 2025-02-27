#!/bin/bash

MODES=("row" "column" "combined")
INPUT_FEATURES=(128)
BATCH_SIZES=(2)
NUM_GPUS=(2)
SEQ_LEN=(128)

LOG_DIR="logs/$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

for mode in "${MODES[@]}"; do
    for feat in "${INPUT_FEATURES[@]}"; do
        for batch in "${BATCH_SIZES[@]}"; do
            for gpus in "${NUM_GPUS[@]}"; do
                for seq_len in "${SEQ_LEN[@]}"; do
                    echo "Running experiment: mode=$mode, features=$feat, batch_size=$batch, gpus=$gpus, seq_len=$seq_len"
                    
                    exp_name="${mode}_f${feat}_b${batch}_g${gpus}_s${seq_len}"
                    log_file="$LOG_DIR/${exp_name}.log"
                    
                    OMP_NUM_THREADS=1 \
                        torchrun \
                            --nproc_per_node=$gpus \
                            main.py \
                            --mode "$mode" \
                            --input-features "$feat" \
                            --output-features "$feat" \
                            --batch-size "$batch" \
                            --seq-len $seq_len \
                            --d-model "$feat" \
                            --warmup 10 \
                            --runs 20 \
                            2>&1 | tee "$log_file"
                        
                    # wait for a while to release gpu resources
                    sleep 2
                done
            done
        done
    done
done

echo "Generating summary..."
{
    echo "Mode,Input Features,Batch Size,GPU Number,Sequence Length,Average Time (ms),Max Time (ms)"
    for log in "$LOG_DIR"/*.log; do
        mode=$(basename "$log" .log | cut -d'_' -f1)
        feat=$(basename "$log" .log | cut -d'_' -f2 | sed 's/f//')
        batch=$(basename "$log" .log | cut -d'_' -f3 | sed 's/b//')
        gpus=$(basename "$log" .log | cut -d'_' -f4 | sed 's/g//')
        seq_len=$(basename "$log" .log | cut -d'_' -f5 | sed 's/s//')
        
        # 分别获取每个rank的时间，并计算它们的平均值
        avg_times=$(grep "Average time:" "$log" | awk '{print $NF}')
        if [ -n "$avg_times" ]; then
            # 计算所有rank中的最大平均时间
            avg_time=$(echo "$avg_times" | awk 'BEGIN {max=0} {if($1>max) max=$1} END {printf "%.6f", max}')
            # 找出所有时间中的最大值作为最大时间
            max_time=$(echo "$avg_times" | awk 'BEGIN {max=0} {if($1>max) max=$1} END {printf "%.6f", max}')
        else
            echo "Warning: No time data found in $log" >&2
            avg_time="N/A"
            max_time="N/A"
        fi
        
        echo "$mode,$feat,$batch,$gpus,$seq_len,$avg_time,$max_time"
    done
} > "$LOG_DIR/summary.csv"

# 显示生成的摘要文件的前几行
echo "Generated summary (first few lines):"
head -n 5 "$LOG_DIR/summary.csv"

echo "Experiments completed. Results saved in $LOG_DIR"
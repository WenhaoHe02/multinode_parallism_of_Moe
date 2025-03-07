MODES=("row" "column" "combined")
INPUT_FEATURES=(64 128 512 1024)
BATCH_SIZES=(1 2 4 8)
NUM_GPUS=(1 2 4 8)
SEQ_LEN=(64 128 256 512)

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
                            script/test_tp.py \
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

PRE_SEQ_LEN=128
CHECKPOINT=law-chatglm-6b-int4-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_predict \
    --validation_file ../../outputs/data/val_law_cases.json \
    --test_file ../../outputs/data/val_law_cases.json \
    --overwrite_cache \
    --prompt_column ajjbqk \
    --response_column pjjg \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4

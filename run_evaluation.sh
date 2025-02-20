CUDA_VISIBLE_DEVICES=6 python evaluation.py \
    --model_name_or_path checkpoints/longgen-soft_sort-rankcse-bert-base-uncased-lr:3e-6-es:25-dw:0.1 \
    --pooler cls \
    --task_set sts \
    --mode test

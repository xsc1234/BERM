#!/bin/bash
#
# This script is for training with updated ann
# training
nohup python -u run_ann_reader_one_gpu.py --model_type reader_one_hot_gelu_equal_cls_add_bm25 \
--model_name_or_path # Initialized retriever, such as the DPR provided by BEIR, or DPR trained by BERM \
--task_name MSMarco \
--triplet \
--data_dir /ir_dataset/msmarco/msmarco/raw_data/ann_data_roberta-base_256/ \
--ann_dir /ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm_log/ann_data/ # The index path generated in the previous step \
--max_seq_length 256 \
--per_gpu_train_batch_size=32 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-6 \
--output_dir /data/xushicheng/ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm_log/ # Path to save the model \
--warmup_steps 5000 \
--logging_steps 100 \
--save_steps 10000 \
--optimizer lamb \
--load_optimizer_scheduler \
--single_warmup > ../log/train_ance_trained_by_berm_log 2>&1 &
#!/bin/bash
#
# This script is for generate ann data for a model in training

nohup python -m torch.distributed.launch --nproc_per_node=2 run_ann_data_gen_reader.py \
--training_dir ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm/ # Always detect whether a new model is generated on the path \
--init_model_dir # Initialized retriever, such as the DPR provided by BEIR, or DPR trained by BERM \
--model_type reader_one_hot_gelu_equal_cls \
--output_dir /ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm/ann_data/ # Output path of index \
--cache_dir /ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm/ann_data/cache/ \
--data_dir /ir_dataset/msmarco/msmarco/raw_data/ann_data_roberta-base_256/ \
--max_seq_length 256 --per_gpu_eval_batch_size 768 --topk_training 200 --negative_sample 20 > ance_trained_by_berm_log 2>&1 &
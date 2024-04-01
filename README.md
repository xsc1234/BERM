# BERM
BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval (ACL 2023)

Paper: https://arxiv.org/abs/2305.11052

```
@inproceedings{BERM,
  author    = {Shicheng Xu and
               Liang Pang and
               Huawei Shen and
               Xueqi Cheng},
  title     = {BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval},
  booktitle = {In Proceedings of the 2023 Conference on ACL},
  year      = {2023},
}
```

## Train with DPR
need about 300000 steps
```
nohup python3 -u ./drivers/run_warmup_reader_one_gpu_origin_model.py \
    --train_model_type reader_one_hot_gelu_equal_cls \
    --model_name_or_path /data/xushicheng/roberta \ #path of roberta
    --task_name MSMarco \
    --do_train \
    --evaluate_during_training \
    --data_dir g/ir_dataset/msmarco/msmarco/raw_data \ #path of msmarco dataset
    --max_seq_length 128 \ 
    --per_gpu_eval_batch_size=256 \
    --per_gpu_train_batch_size=32 \
    --learning_rate 2e-5  \
    --logging_steps 1000  \
    --num_train_epochs 2.0  \
    --output_dir /BERM/out/dpr_trained_by_berm \ # Path to save the model
    --warmup_steps 1000  \
    --overwrite_output_dir \
    --save_steps 10000 \
    --gradient_accumulation_steps 1 \
    --expected_train_size 35000000 \
    --logging_steps_per_eval 10000000 \
    --fp16 \
    --optimizer lamb \
    --log_dir ../log/tensorboard/dpr > /dpr_trained_by_berm_log 2>&1 &
```

## Train with ANCE

### Step 1. preprocess msmarco:
```
sh ./cmds/preprocess_21.sh
```

### Step 2. Use the trained DPR model by BERM to initialize the retriever and generate the initial index
```
nohup python -m torch.distributed.launch --nproc_per_node=2 ./drivers/run_ann_data_gen_reader.py \
--training_dir ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm/ # Always detect whether a new model is generated on the path \
--init_model_dir # Initialized retriever, such as the DPR provided by BEIR, or DPR trained by BERM \
--model_type reader_one_hot_gelu_equal_cls \
--output_dir /ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm/ann_data/ # Output path of index \
--cache_dir /ir_dataset/msmarco/msmarco/raw_data/ance_trained_by_berm/ann_data/cache/ \
--data_dir /ir_dataset/msmarco/msmarco/raw_data/ann_data_roberta-base_256/ \
--max_seq_length 256 --per_gpu_eval_batch_size 768 --topk_training 200 --negative_sample 20 > ance_trained_by_berm_log 2>&1 &
```

### Step 3. After the initial index is generated, start to train the model with BERM

```
nohup python -u ./drivers/run_ann_reader_one_gpu.py --model_type reader_one_hot_gelu_equal_cls_add_bm25 \
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
```

In the training process of Step 3, Step 2 should keep running (parallel). When Step 2 detects the new model got from Step 3, it will use the new model to generate the new index, and then Step 3 will use the new index to continue training.


#!/bin/bash

is_causal=True
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="beomi"
model_type="Llama-3-KoEn-8B"
left_padding=False
quantization_type="origin"
peft_type="origin"
data_max_length=256
target_max_length=256
precision="bf16"
batch_size=32
epochs="3 4"

for epoch in $epochs
do
    python main.py mode=predict \
        is_causal=$is_causal \
        is_preprocessed=$is_preprocessed \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        left_padding=$left_padding \
        quantization_type=$quantization_type \
        peft_type=$peft_type \
        data_max_length=$data_max_length \
        target_max_length=$target_max_length \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done

for epoch in $epochs
do
    python merge_predictions.py \
        is_causal=$is_causal \
        is_preprocessed=$is_preprocessed \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        left_padding=$left_padding \
        quantization_type=$quantization_type \
        peft_type=$peft_type \
        data_max_length=$data_max_length \
        target_max_length=$target_max_length \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done

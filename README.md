# Reame
This repo is the source code for paper Biomedical named entity normalization via interaction-based synonym marginalization

## Dataset
you can download dataset from [BioSyn](https://github.com/dmis-lab/BioSyn#datasets)

## Train
```bash
model_type="sapbert"
epoch=10
pair_weight=5
bert_learning_rate=1e-5

if [ $model_type == "biobert" ]
then 
    model_path="./public_model/biobert-base-cased-v1.1"
else
    model_path="./public_model/SapBERT-from-PubMedBERT-fulltext"
fi

output_dir=./outputs/$model_type+$random_seed/$dataset_name/train+dev/

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

echo $model_path "," $dataset_name "," $epoch, "," $output_dir

CUDA_VISIBLE_DEVICES=$1 python -u train.py \
    --model_name_or_path ${model_path} \
    --train_dictionary_path ./data/${dataset_name}/train_dictionary.txt \
    --train_dir ./data/${dataset_name}/processed_traindev \
    --output_dir ${output_dir} \
    --use_cuda \
    --topk 20 \
    --epoch ${epoch} \
    --train_batch_size 16 \
    --bert_learning_rate $bert_learning_rate \
    --other_learning_rate 1e-4 \
    --max_length 25 \
    --weight_decay 0.01 \
    --save_checkpoint_all \
    --add_dense \
    --add_sparse \
    --pair_weight=$pair_weight \
    --add_pair_atten \
    --attention_score_mode="dot" 
```
## Evaluation
We shared our pretrained model at https://drive.google.com/drive/folders/1yhnBt8M42suc5u7ZCI9yHAyao7MQF0Km?usp=sharing

```bash
output_dir="./checkpoints/ncbi-disease/"
dataset_dir="./data/ncbi-disease/"

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --model_name_or_path $output_dir \
    --model_checkpoint $output_dir/ep.pth \
    --dictionary_path $dataset_dir/test_dictionary.txt \
    --data_dir $dataset_dir/processed_test \
    --output_dir $output_dir \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions
```

dataset_name="bc5cdr-chemical"

output_dir="./checkpoints/"$dataset_name"/"
dataset_dir="./data/"$dataset_name"/"

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
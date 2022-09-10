# 验证集选最优epoch再测测试集
for model_type in "sapbert"
do
    for dataset_name in "bc5cdr-chemical" 
    do
        for random_seed in 10307 10926 12293 1406 6728
        do 
            for epoch in 1 2 3 4 5 6 7 8 9 10
            do
                output_dir=./outputs-chem/$model_type+$random_seed/$dataset_name/train+dev/checkpoint_$epoch  # 不能有/
                echo "evaluating" $output_dir

                CUDA_VISIBLE_DEVICES=$1 python eval.py \
                    --model_name_or_path $output_dir \
                    --model_checkpoint ./outputs-chem/$model_type+$random_seed/$dataset_name/train+dev/ep-${epoch}.pth \
                    --dictionary_path ./data/$dataset_name/test_dictionary.txt \
                    --data_dir ./data/$dataset_name/processed_test \
                    --output_dir $output_dir \
                    --use_cuda \
                    --topk 20 \
                    --max_length 25 \
                    --save_predictions
            done
        done
    done
done
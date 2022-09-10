import argparse
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm
import torch
from utils import (
    evaluate
)
from src.biosyn import (
    DictionaryDataset,
    QueryDataset,
    BioSyn,
    RerankNet
)
LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='BioSyn evaluation')

    # Required
    parser.add_argument('--model_name_or_path', required=True, help='Directory for model')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')

    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--topk',  type=int, default=20)
    parser.add_argument('--score_mode',  type=str, default='hybrid', help='hybrid/dense/sparse')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")

    parser.add_argument('--model_checkpoint', required=True, help='Directory for model')

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)
    
    args = parser.parse_args()
    return args
    
def init_logging(args):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    logfile_path = os.path.join(args.output_dir, str(datetime.now()).split(".")[0].replace(" ", "-")+".txt")
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(fmt)

    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)

def load_dictionary(dictionary_path): 
    dictionary = DictionaryDataset(
        dictionary_path = dictionary_path
    )
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data
                
def main(args):
    init_logging(args)
    print(args.__dict__)

    with open(os.path.join(os.path.dirname(args.output_dir), "args.json")) as f:
        train_args = json.load(f)

    # load dictionary and data
    eval_dictionary = load_dictionary(dictionary_path=args.dictionary_path)
    eval_queries = load_queries(
        data_dir=args.data_dir,
        filter_composite=args.filter_composite,
        filter_duplicate=args.filter_duplicate
    )

    biosyn = BioSyn(
        max_length=args.max_length,
        use_cuda=args.use_cuda
    )
    biosyn.load_model(
        model_name_or_path=args.model_name_or_path,
    )

    model = RerankNet(
        encoder=biosyn.encoder,
        sparse_weight=biosyn.sparse_weight,
        add_sparse=train_args["add_sparse"],
        add_dense=train_args["add_dense"],
        add_option_atten=train_args["add_option_atten"],
        add_pair_atten=train_args["add_pair_atten"],
        pair_weight=train_args["pair_weight"],
        attention_score_mode=train_args["attention_score_mode"]
    )
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.cuda()
    
    result_evalset = evaluate(
        biosyn=biosyn,
        model=model, 
        eval_dictionary=eval_dictionary,
        eval_queries=eval_queries,
        topk=args.topk,
        max_length=train_args["max_length"]
    )
    
    LOGGER.info("acc@1={}".format(result_evalset['acc1']))
    
    if args.save_predictions:
        if "test" in args.data_dir:
            output_file_name = "predictions_test.json"
        else:
            output_file_name = "predictions_eval.json"
        output_file = os.path.join(args.output_dir, output_file_name)
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)

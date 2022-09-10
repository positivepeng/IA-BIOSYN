import numpy as np
import torch
import argparse
import logging
from datetime import datetime
import time
import torch.optim as optim
import os
import json
import random
from utils import (
    evaluate
)
from tqdm import tqdm
from src.biosyn import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    BioSyn
)

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    # Required
    parser.add_argument('--model_name_or_path', required=True,
                        help='Directory for pretrained model')
    parser.add_argument('--train_dictionary_path', type=str, required=True,
                    help='train dictionary path')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--draft',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--bert_learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--other_learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_sparse_weight',
                        default=0, type=float)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")

    parser.add_argument("--pair_weight", default=1, type=float, help="weight of pair representation similarity score")
    parser.add_argument('--add_sparse', action="store_true", help="add S_sparse to score")
    parser.add_argument('--add_dense', action="store_true", help="add S_dense to score")
    parser.add_argument('--add_option_atten', action="store_true")
    parser.add_argument('--add_pair_atten', action="store_true")
    parser.add_argument('--attention_score_mode', type=str, default="dot")

    args = parser.parse_args()
    return args

def init_logging(args):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logfile_path = os.path.join(args.output_dir, str(datetime.now()).split(".")[0].replace(" ", "-")+".txt")
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(fmt)

    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate, filter_cuiless):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    filter_cuiless : bool
        filter samples with cuiless
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate,
        filter_cuiless=filter_cuiless
    )
    
    return dataset.data

def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging(args)
    init_seed(args.seed)
    print(args.__dict__)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        print(json.dumps(args.__dict__, indent=4))
        json.dump(args.__dict__, f, indent=4)
        
    # load dictionary and queries
    train_dictionary = load_dictionary(dictionary_path=args.train_dictionary_path)
    train_queries = load_queries(
        data_dir = args.train_dir, 
        filter_composite=True,
        filter_duplicate=True,
        filter_cuiless=True
    )

    if args.draft:
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        args.output_dir = args.output_dir + "_draft"
        
    # filter only names
    names_in_train_dictionary = train_dictionary[:,0]
    names_in_train_queries = train_queries[:,0]

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = BioSyn(
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        initial_sparse_weight=args.initial_sparse_weight
    )
    biosyn.init_sparse_encoder(corpus=names_in_train_dictionary)
    biosyn.load_dense_encoder(
        model_name_or_path=args.model_name_or_path
    )

    model = RerankNet(
        encoder=biosyn.encoder,
        sparse_weight=biosyn.sparse_weight,
        add_sparse=args.add_sparse,
        add_dense=args.add_dense,
        add_option_atten=args.add_option_atten,
        add_pair_atten=args.add_pair_atten,
        pair_weight=args.pair_weight,
        attention_score_mode=args.attention_score_mode
    )
    model.cuda()

    bert_parameters, other_parameters = [], []
    for name, p in model.named_parameters():
        if name.startswith("encoder"):
            bert_parameters.append(p)
        elif name != "sparse_weight":
            other_parameters.append(p)
    parameter_groups = [
        {'params': bert_parameters, "lr":args.bert_learning_rate, "weight_decay":args.weight_decay},            
        {'params': other_parameters, "lr":args.other_learning_rate, "weight_decay":args.weight_decay},      
        {'params': model.sparse_weight, 'lr': 0.01, 'weight_decay': 0}
    ]
    model.optimizer = optim.Adam(parameter_groups)

    # embed sparse representations for query and dictionary
    # Important! This is one time process because sparse represenation never changes.
    LOGGER.info("Sparse embedding")
    train_query_sparse_embeds = biosyn.embed_sparse(names=names_in_train_queries)
    train_dict_sparse_embeds = biosyn.embed_sparse(names=names_in_train_dictionary)
    train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=train_query_sparse_embeds, 
        dict_embeds=train_dict_sparse_embeds
    )
    train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=train_sparse_score_matrix, 
        topk=args.topk
    )

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = biosyn.get_dense_tokenizer(), 
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs,
        topk = args.topk, 
        d_ratio=args.dense_ratio,
        max_length=args.max_length
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    start = time.time()
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))

        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)

        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        train_query_dense_embeds = biosyn.embed_dense(names=names_in_train_queries, show_progress=True)
        train_dict_dense_embeds = biosyn.embed_dense(names=names_in_train_dictionary, show_progress=True)
        train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_dense_embeds, 
            dict_embeds=train_dict_dense_embeds
        )
        train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=train_dense_score_matrix, 
            topk=args.topk
        )
        # replace dense candidates in the train_set
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        # train
        train_loss = train(args, data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        LOGGER.info("sparse_weight={:.4f}".format(model.sparse_weight.item()))
        
        # save model every epoch
        model_save_path = os.path.join(args.output_dir, "ep-{}.pth".format(epoch))
        torch.save(model.state_dict(), model_save_path)
        
        # save model last epoch
        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

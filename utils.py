import csv
import json
import numpy as np
import pdb
from tqdm import tqdm
import torch

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(biosyn, model, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    encoder = biosyn.get_dense_encoder()
    tokenizer = biosyn.get_dense_tokenizer()
    sparse_encoder = biosyn.get_sparse_encoder()
    sparse_weight = biosyn.get_sparse_weight().item() # must be scalar value
    
    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:,0], show_progress=True)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:,0], show_progress=True)
    
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        
        dict_mentions = []
        for mention in mentions:
            mention_sparse_embeds = biosyn.embed_sparse(names=np.array([mention]))
            mention_dense_embeds = biosyn.embed_dense(names=np.array([mention]))
            
            # get score matrix
            sparse_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sparse_embeds, 
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dict_dense_embeds
            )
            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            else:
                raise NotImplementedError()
            
            candidate_idxs_sparse = biosyn.retrieve_candidate(
                score_matrix = sparse_score_matrix, 
                topk = topk//2
            )
            candidate_idxs_dense = biosyn.retrieve_candidate(
                score_matrix = dense_score_matrix, 
                topk = topk//2
            )

            # biosyn的排序结果
            # candidate_idxs = biosyn.retrieve_candidate(
            #     score_matrix = score_matrix, 
            #     topk = topk
            # )

            candidate_idxs = np.concatenate([candidate_idxs_sparse, candidate_idxs_dense], -1)

            
            np_candidates = eval_dictionary[candidate_idxs].squeeze()

            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'cui':np_candidate[1],
                    'label':check_label(np_candidate[1],golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result


def predict_topk_match(biosyn, model, eval_dictionary, all_eval_queries, topk, max_length, dense_ratio=0.5):
    print("eval total", len(all_eval_queries))

    tokenizer = biosyn.get_dense_tokenizer()

    eval_dictionary_names = eval_dictionary[:, 0]
    eval_dictionary_cuis = eval_dictionary[:, 1]
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary_names, show_progress=False)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary_names, show_progress=True)
    
    all_queries = []

    number_each_iter = 1000
    iter_num = len(all_eval_queries) // number_each_iter + 1    
    for iter_idx in range(iter_num):
        eval_queries = all_eval_queries[iter_idx*number_each_iter: (iter_idx+1)*number_each_iter]  # 防止爆内存

        if len(eval_queries) == 0:
            break

        eval_dataset = []
        for eval_query in tqdm(eval_queries, total=len(eval_queries)):
            # 有可能有多个mention 对应 多个CUI
            mentions = eval_query[0].replace("+","|").split("|")
            golden_cui = eval_query[1].replace("+","|")
            for mention in mentions:
                eval_dataset.append(
                    {
                        "mention": mention,  
                        "golden_cui": golden_cui
                    }
                )
        print("query size={} dataset size={}".format(len(eval_queries), len(eval_dataset)))

        eval_mentions = [d["mention"] for d in eval_dataset]
        query_sparse_embeds = biosyn.embed_sparse(names=eval_mentions, show_progress=False)
        sparse_score_matrix = biosyn.get_score_matrix(
            query_embeds = query_sparse_embeds,
            dict_embeds = dict_sparse_embeds
        )
        query_dense_embeds = biosyn.embed_dense(names=eval_mentions, show_progress=True)
        dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=query_dense_embeds, 
            dict_embeds=dict_dense_embeds
        )
        dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=dense_score_matrix, 
            topk=topk
        )
        sparse_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=sparse_score_matrix, 
            topk=topk
        )

        n_dense = int(topk * dense_ratio)
        n_sparse = topk - n_dense
        all_sparse_scores = []
        all_candidate_names = []
        all_candidate_cuis = []
        all_mention_candidate_pairs = []
        for mention_idx in range(len(eval_mentions)):
            temp_sparse_candidate_idxs = sparse_candidate_idxs[mention_idx]
            temp_dense_candidate_idxs = dense_candidate_idxs[mention_idx]

            topk_candidate_idx = temp_sparse_candidate_idxs[:n_sparse]
            for d_idx in temp_dense_candidate_idxs:
                if len(topk_candidate_idx) >= topk:
                    break
                if d_idx not in topk_candidate_idx:
                    topk_candidate_idx = np.append(topk_candidate_idx, d_idx)

            assert len(topk_candidate_idx) == topk
            assert len(topk_candidate_idx) == len(set(topk_candidate_idx))  # 无重复
            
            candidate_names = [eval_dictionary_names[candidate_idx].tolist() for candidate_idx in topk_candidate_idx]
            candidate_cuis = [eval_dictionary_cuis[candidate_idx].tolist() for candidate_idx in topk_candidate_idx]
            sparse_scores = [sparse_score_matrix[mention_idx][candidate_idx].tolist() for candidate_idx in topk_candidate_idx]
            mention_candidate_pairs = [[eval_mentions[mention_idx], candidate_name] for candidate_name in candidate_names]
            
            all_candidate_names.extend(candidate_names)
            all_candidate_cuis.extend(candidate_cuis)
            all_sparse_scores.extend(sparse_scores)
            all_mention_candidate_pairs.extend(mention_candidate_pairs)

        
        pred_scores = []

        batch_size = 32
        batch_num = len(eval_dataset) // batch_size + 1
        for batch_idx in tqdm(range(batch_num)):
            start_idx = batch_size * batch_idx
            end_idx = min(len(eval_dataset), batch_size*(batch_idx+1))

            batch_query_token = eval_mentions[start_idx:end_idx]
            batch_candidate_token = all_candidate_names[start_idx*topk:end_idx*topk]
            batch_candidate_cuis = all_candidate_cuis[start_idx*topk:end_idx*topk]
            batch_pair_tokens = all_mention_candidate_pairs[start_idx*topk:end_idx*topk]
            batch_sparse_scores = all_sparse_scores[start_idx*topk:end_idx*topk]

            batch_mention_tokens = tokenizer(batch_query_token, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            batch_candidate_tokens = tokenizer(batch_candidate_token, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            batch_mention_candidate_tokens = tokenizer(batch_pair_tokens, max_length=max_length*2, padding='max_length', truncation=True, return_tensors='pt')

            for k in batch_mention_tokens:
                batch_mention_tokens[k] = batch_mention_tokens[k].cuda()
            for k in batch_candidate_tokens:
                batch_candidate_tokens[k] = batch_candidate_tokens[k].reshape(-1, topk, max_length).cuda()
            for k in batch_mention_candidate_tokens:
                batch_mention_candidate_tokens[k] = batch_mention_candidate_tokens[k].reshape(-1, topk, 2*max_length).cuda()
            batch_sparse_scores = torch.tensor(batch_sparse_scores).reshape(-1, topk).cuda()
            batch_candidate_cuis = np.array(batch_candidate_cuis).reshape(-1, topk)

            model.eval()
            with torch.no_grad():
                batch_x = (batch_mention_tokens, batch_candidate_tokens, batch_mention_candidate_tokens, batch_sparse_scores)
                batch_pred_score = model(batch_x)
                batch_pred_score = batch_pred_score.cpu().numpy().tolist()
                pred_scores.extend(batch_pred_score)

        all_candidate_names = np.array(all_candidate_names).reshape(-1, topk)
        all_candidate_cuis = np.array(all_candidate_cuis).reshape(-1, topk)

        assert len(pred_scores) == len(eval_dataset) == len(all_candidate_names) == len(all_candidate_cuis)
        
        
        curr_pos = 0
        for eval_query in tqdm(eval_queries, total=len(eval_queries)):
            mentions = eval_query[0].replace("+","|").split("|")
            golden_cui = eval_query[1].replace("+","|")

            dict_mentions = []
            for mention in mentions:
                assert eval_dataset[curr_pos]["mention"] in eval_query[0]
                pred_score = pred_scores[curr_pos]
                pred_score_argsort = (np.array(pred_score)*-1).argsort()
                candidate_names = all_candidate_names[curr_pos][pred_score_argsort]
                candidate_cuis = all_candidate_cuis[curr_pos][pred_score_argsort]
                np_candidates = [
                    [name, cui] for name, cui in zip(candidate_names, candidate_cuis)
                ]

                dict_candidates = []
                for np_candidate in np_candidates:
                    dict_candidates.append({
                        'name':np_candidate[0],
                        'cui':np_candidate[1],
                        'label':check_label(np_candidate[1], golden_cui)
                    })
                dict_mentions.append({
                    'mention':mention,
                    'golden_cui':golden_cui, # golden_cui can be composite cui
                    'candidates':dict_candidates
                })

                curr_pos += 1

            all_queries.append({
                'mentions':dict_mentions
            })

    print("len queries={}".format(len(all_queries)))

    result = {
        'queries':all_queries
    }

    return result


def evaluate(biosyn, model, eval_dictionary, eval_queries, topk, max_length, score_mode='hybrid'):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : str
        hybrid, dense, sparse
    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk_match(biosyn, model, eval_dictionary, eval_queries, topk, max_length)
    result = evaluate_topk_acc(result)
    
    return result
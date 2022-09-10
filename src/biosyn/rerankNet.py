import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)

class RerankNet(nn.Module):
    def __init__(self, 
                encoder, 
                sparse_weight, 
                add_sparse=False, 
                add_dense=False, 
                add_option_atten=False, 
                add_pair_atten=False, 
                pair_weight=1, 
                attention_score_mode="dot"):

        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.sparse_weight = sparse_weight
        
        self.add_sparse = add_sparse
        self.add_dense = add_dense
        self.add_pair_atten = add_pair_atten
        self.add_option_atten = add_option_atten
        self.attention_score_type = attention_score_mode
        self.pair_weight = pair_weight

        if add_option_atten:
            self.match_weight = pair_weight
            self.match_layer = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(768*4, 768*2),
                nn.ReLU(),
                nn.Linear(768*2, 1)
            )
            self.option_attention = OptionAtten(score_mode=attention_score_mode)
        if add_pair_atten:
            self.pair_attention = OptionAtten(score_mode=attention_score_mode)

        self.pair2score = nn.Linear(768, 1)
        self.criterion = marginal_nll

    def forward(self, x):
        query_token, candidate_tokens, query_candidate_tokens, sparse_scores = x
        
        batch_size, topk, max_length = candidate_tokens['input_ids'].shape
        _, _, pair_max_length = query_candidate_tokens["input_ids"].shape

        score = torch.zeros((batch_size, topk)).cuda()

        # sparse score
        if self.add_sparse:
            score += self.sparse_weight * sparse_scores

        # pair score 
        pair_embed = self.encoder(
                input_ids=query_candidate_tokens['input_ids'].reshape(-1, pair_max_length),
                token_type_ids=query_candidate_tokens['token_type_ids'].reshape(-1, pair_max_length),
                attention_mask=query_candidate_tokens['attention_mask'].reshape(-1, pair_max_length)
            )
        pair_embed = pair_embed[0][:, 0].reshape(batch_size, topk, -1)

        if self.add_pair_atten:
            pair_embed = self.pair_attention(pair_embed)
        
        pair_score = self.pair2score(pair_embed).squeeze(-1)
        score += pair_score * self.pair_weight

        # dense score
        if self.add_dense:
            query_embed = self.encoder(
                input_ids=query_token['input_ids'].squeeze(1),
                token_type_ids=query_token['token_type_ids'].squeeze(1),
                attention_mask=query_token['attention_mask'].squeeze(1)
            )
            query_embed = query_embed[0][:,0].unsqueeze(1) # query : [batch_size, 1, hidden]
            
            candidate_embeds = self.encoder(
                input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
                token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, max_length),
                attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length)
            )
            candidate_embeds = candidate_embeds[0][:,0].reshape(batch_size, topk, -1) # [batch_size, topk, hidden]
            
            candidate_d_score = torch.bmm(query_embed, candidate_embeds.permute(0,2,1)).squeeze(1)
            score += candidate_d_score

            if self.add_option_atten:
                candidate_embeds = self.option_attention(candidate_embeds)
                # option-atten based on text matching
                query_embed_expand = query_embed.expand(batch_size, topk, -1)        
                u, v = query_embed_expand, candidate_embeds
                uv = torch.cat([u, v, u-v, u*v], dim=-1)
                match_score = self.match_layer(uv).squeeze(-1)
                score += match_score * self.match_weight
                
        return score

        # if self.add_sparse:
        #     # sparse_scores /= torch.norm(sparse_scores, 2, dim=-1, keepdim=True)  # 不加
        #     # sparse_scores /= torch.max(sparse_scores, dim=-1, keepdim=True)[0]
        #     score += self.sparse_weight * sparse_scores  

      

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        if isinstance(candidates, dict):
            for k in candidates:
                _, _, max_length = candidates[k].shape
                candidates[k] = candidates[k].contiguous().view(-1, max_length)
        else:
            _, _, max_length = candidates.shape
            candidates = candidates.contiguous().view(-1, max_length)

        return candidates

    def get_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss

    # def get_embeddings(self, mentions, batch_size=1024):
    #     """
    #     Compute all embeddings from mention tokens.
    #     """
    #     embedding_table = []
    #     with torch.no_grad():
    #         for start in tqdm(range(0, len(mentions), batch_size)):
    #             end = min(start + batch_size, len(mentions))
    #             batch = mentions[start:end]
    #             batch_embedding = self.vectorizer(batch)
    #             batch_embedding = batch_embedding.cpu()
    #             embedding_table.append(batch_embedding)
    #     embedding_table = torch.cat(embedding_table, dim=0)
    #     return embedding_table

class OptionAtten(nn.Module):
    def __init__(self, score_mode="dot"):
        super(OptionAtten, self).__init__()

        self.score_mode = score_mode

        if score_mode == "dcmn":
            self.w5 = nn.Linear(768, 768, bias=False)

            self.w7 = nn.Linear(768, 768, bias=False)
            self.w8 = nn.Linear(768, 768)
            
    
    def forward(self, options, ):
        batch_size, option_num, hidden = options.size()

        if self.score_mode == "dot":
            mask = 1 - torch.from_numpy(np.eye(option_num)).float().cuda()

            att_matrix = options.bmm(torch.transpose(options, 1, 2))
            att_matrix /= 28

            att_weight = torch.nn.functional.softmax(att_matrix, dim=-1)
            att_vec = att_weight.bmm(options)
            att_options = att_vec.view(batch_size, option_num, -1)
        
        elif self.score_mode == "dcmn":
            transed_options = self.w5(options)
            G = nn.Softmax(dim=-1)(torch.bmm(transed_options, torch.transpose(options, 1, 2)))
            H = nn.ReLU()(torch.bmm(G, options))

            mask = torch.from_numpy(1 - np.eye(option_num)).float().cuda() / (option_num - 1)
            mask = torch.stack([mask for _ in range(batch_size)])

            other_option_mean = torch.bmm(mask, H)

            g = nn.Sigmoid()(self.w7(other_option_mean) + self.w8(options))

            att_options = g * H + (1 - g) * other_option_mean

        return att_options



# class RerankNet(nn.Module):
#     def __init__(self, encoder, learning_rate, weight_decay, sparse_weight, use_cuda):

#         LOGGER.info("RerankNet! learning_rate={} weight_decay={} sparse_weight={} use_cuda={}".format(
#             learning_rate,weight_decay,sparse_weight,use_cuda
#         ))
#         super(RerankNet, self).__init__()
#         self.encoder = encoder
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.use_cuda = use_cuda
#         self.sparse_weight = sparse_weight
#         self.optimizer = optim.Adam([
#             {'params': self.encoder.parameters()},
#             {'params' : self.sparse_weight, 'lr': 0.01, 'weight_decay': 0}], 
#             lr=self.learning_rate, weight_decay=self.weight_decay
#         )
        
#         self.criterion = marginal_nll
        
#     def forward(self, x):
#         """
#         query : (N, h), candidates : (N, topk, h)

#         output : (N, topk)
#         """
#         query_token, candidate_tokens, candidate_s_scores = x
#         batch_size, topk, max_length = candidate_tokens['input_ids'].shape

#         if self.use_cuda:
#             candidate_s_scores = candidate_s_scores.cuda()
#             query_token['input_ids'] = query_token['input_ids'].to('cuda')
#             query_token['token_type_ids'] = query_token['token_type_ids'].to('cuda')
#             query_token['attention_mask'] = query_token['attention_mask'].to('cuda')
#             candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to('cuda')
#             candidate_tokens['token_type_ids'] = candidate_tokens['token_type_ids'].to('cuda')
#             candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to('cuda')


#         # dense embed for query and candidates
#         query_embed = self.encoder(
#             input_ids=query_token['input_ids'].squeeze(1),
#             token_type_ids=query_token['token_type_ids'].squeeze(1),
#             attention_mask=query_token['attention_mask'].squeeze(1)
#         )
#         query_embed = query_embed[0][:,0].unsqueeze(1) # query : [batch_size, 1, hidden]
        
#         candidate_embeds = self.encoder(
#             input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
#             token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, max_length),
#             attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length)
#         )
#         candidate_embeds = candidate_embeds[0][:,0].reshape(batch_size, topk, -1) # [batch_size, topk, hidden]
        
#         # score dense candidates
#         candidate_d_score = torch.bmm(query_embed, candidate_embeds.permute(0,2,1)).squeeze(1)
#         score = self.sparse_weight * candidate_s_scores + candidate_d_score
#         return score

#     def reshape_candidates_for_encoder(self, candidates):
#         """
#         reshape candidates for encoder input shape
#         [batch_size, topk, max_length] => [batch_size*topk, max_length]
#         """
#         _, _, max_length = candidates.shape
#         candidates = candidates.contiguous().view(-1, max_length)
#         return candidates

#     def get_loss(self, outputs, targets):
#         if self.use_cuda:
#             targets = targets.cuda()
#         loss = self.criterion(outputs, targets)
#         return loss

#     def get_embeddings(self, mentions, batch_size=1024):
#         """
#         Compute all embeddings from mention tokens.
#         """
#         embedding_table = []
#         with torch.no_grad():
#             for start in tqdm(range(0, len(mentions), batch_size)):
#                 end = min(start + batch_size, len(mentions))
#                 batch = mentions[start:end]
#                 batch_embedding = self.vectorizer(batch)
#                 batch_embedding = batch_embedding.cpu()
#                 embedding_table.append(batch_embedding)
#         embedding_table = torch.cat(embedding_table, dim=0)
#         return embedding_table


def marginal_nll(score, target):
    """
    sum all scores among positive samples
    """
    predict = F.softmax(score, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)                   # sum all positive scores
    loss = loss[loss > 0]                     # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
    loss = -torch.log(loss)                   # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()                     # will return zero loss
    else:
        loss = loss.mean()
    return loss
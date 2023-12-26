import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained('pretrained_models/llama2-7b-hf')

class PartiallyTrainedEmbedding1(nn.Embedding):
    def __init__(self, new_embedding_nums, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, padding_idx, **kwargs)
        self.new_weight = nn.Parameter(torch.randn(new_embedding_nums, embedding_dim), requires_grad=True)
        torch.nn.init.trunc_normal_(self.new_weight.data,mean=0.0,std=0.02)
    def forward(self, input):
        # import pdb
        # pdb.set_trace()
        mask = input >= self.num_embeddings
        # You may want to optimize it, you could probably get away without copy, though
        # I'm not currently sure how
        pretrained_batch = input.clone()
        pretrained_batch[mask] = 0
        embedded_batch = F.embedding(
                    pretrained_batch, self.weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)
        
        # Every token without representation has to be brought into appropriate range
        input -= self.num_embeddings
        # Zero out the ones which already have pretrained embedding
        input[~mask] = 0
        non_pretrained_embedded_batch = F.embedding(
                    input, self.new_weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)
        # And finally change appropriate tokens from placeholder embedding created by
        # pretrained into trainable embeddings.
        embedded_batch[mask] = non_pretrained_embedded_batch[mask]
        return embedded_batch
        # return F.embedding(
        #             input, self.weight, self.padding_idx, self.max_norm,
        #             self.norm_type, self.scale_grad_by_freq, self.sparse)

class PartiallyTrainedEmbedding2(nn.Embedding):
    def __init__(self, new_embedding_nums, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, padding_idx, **kwargs)
        self.new_weight = nn.Parameter(torch.randn(new_embedding_nums, embedding_dim), requires_grad=True)
        torch.nn.init.trunc_normal_(self.new_weight.data,mean=0.0,std=0.02)
    def forward(self, input):
        # import pdb
        # pdb.set_trace()
        # mask = input >= self.num_embeddings
        # # You may want to optimize it, you could probably get away without copy, though
        # # I'm not currently sure how
        # pretrained_batch = input.clone()
        # pretrained_batch[mask] = 0
        # embedded_batch = F.embedding(
        #             pretrained_batch, self.weight, self.padding_idx, self.max_norm,
        #             self.norm_type, self.scale_grad_by_freq, self.sparse)
        
        # # Every token without representation has to be brought into appropriate range
        # input -= self.num_embeddings
        # # Zero out the ones which already have pretrained embedding
        # input[~mask] = 0
        # non_pretrained_embedded_batch = F.embedding(
        #             input, self.new_weight, self.padding_idx, self.max_norm,
        #             self.norm_type, self.scale_grad_by_freq, self.sparse)
        # # And finally change appropriate tokens from placeholder embedding created by
        # # pretrained into trainable embeddings.
        # embedded_batch[mask] = non_pretrained_embedded_batch[mask]
        # return embedded_batch
        return F.embedding(
                    input, self.weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)



text = 'hello, my good brother'
tokens = tokenizer(text, return_tensors="pt").input_ids[:,1:]
model_1 = PartiallyTrainedEmbedding1(10,32000,256)
model_2 = PartiallyTrainedEmbedding2(10,32000,256)
model_2_stat_dict = model_2.state_dict()
model_1.load_state_dict(model_2_stat_dict)

import pdb
pdb.set_trace()
out1 = model_1(tokens)
tokens = tokenizer(text, return_tensors="pt").input_ids[:,1:]

out2 = model_2(tokens)

print(torch.allclose(out1, out2))

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, LlamaForCausalLM, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig, WhisperConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.models.whisper.modeling_whisper import _compute_mask_indices

from transformers import Wav2Vec2Model, AutoProcessor

try:
    from .configuration_blsp import BlspConfig
    from .modeling_whisper_encoder import WhisperEncoder
    from .moe_layer import MoELayer
except:
    from configuration_blsp import BlspConfig
    from modeling_whisper_encoder import WhisperEncoder
    from moe_layer import MoELayer

def lengths_to_padding_mask(lens, max_lens=None):
    if max_lens is None:
        max_lens = torch.max(lens).item()
    bsz = lens.size(0)
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        moe: False
    ):
        super(Adapter, self).__init__()

        self.moe = moe
        if not moe:
            self.activation = nn.GELU()
            self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
            self.fc2 = nn.Linear(mid_dim, in_dim, bias=False)
        else:
            self.expert1 = nn.Linear(in_dim, mid_dim, bias=False)
            self.moe_layer1 = MoELayer(hidden_size=in_dim, num_experts=4, expert=self.expert1, 
                     route_method='gate-token', vocab_size=None, hash_list=None)
            self.expert2 = nn.Linear(mid_dim, in_dim, bias=False)
            self.moe_layer2 = MoELayer(hidden_size=mid_dim, num_experts=4, expert=self.expert2, 
                     route_method='gate-token', vocab_size=None, hash_list=None)
            self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        if not self.moe:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            return residual + x, 0.0
        else:
            x, balance_loss1, gate_load1 = self.moe_layer1(x, None, None)
            x = self.activation(x)
            x, balance_loss2, gate_load2 = self.moe_layer2(x, None, None)
        return x, balance_loss1 + balance_loss2
            
class PartiallyTrainedEmbedding(nn.Embedding):
    def __init__(self, new_embedding_nums, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, padding_idx, **kwargs)
        self.new_weight = nn.Parameter(torch.randn(new_embedding_nums, embedding_dim), requires_grad=True)
        torch.nn.init.trunc_normal_(self.new_weight.data,mean=0.0,std=0.02)
        
    def forward(self, input):

        mask = input >= self.num_embeddings
        # You may want to optimize it, you could probably get away without copy, though
        # I'm not currently sure how
        pretrained_batch = input.clone()
        pretrained_batch[mask] = 0
        
        embedded_batch = F.embedding(
                    pretrained_batch, self.weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)
        
        # Every token without representation has to be brought into appropriate range
        non_pretrained_batch = input.clone()
        non_pretrained_batch -= self.num_embeddings
        # Zero out the ones which already have pretrained embedding
        non_pretrained_batch[~mask] = 0
        non_pretrained_embedded_batch = F.embedding(
                    non_pretrained_batch, self.new_weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)
        # And finally change appropriate tokens from placeholder embedding created by
        # pretrained into trainable embeddings.
        embedded_batch[mask] = non_pretrained_embedded_batch[mask]
        return embedded_batch

class PartiallyTrainedLMHead(nn.Linear):
    def __init__(self, new_embedding_nums, embedding_dim, num_embeddings, **kwargs):
        super().__init__(embedding_dim, num_embeddings, **kwargs)
        self.new_weight = nn.Parameter(torch.randn(new_embedding_nums, embedding_dim), requires_grad=True)
        self.new_bias =  self.register_parameter('new_bias', None)
        torch.nn.init.trunc_normal_(self.new_weight.data,mean=0.0,std=0.02)
    def forward(self, input):
        x1 = F.linear(input, self.weight, self.bias)
        x2 = F.linear(input, self.new_weight, self.new_bias)
        return torch.cat([x1, x2], dim=-1)

class BlspModel(PreTrainedModel):
    config_class = BlspConfig
    base_model_prefix = "blsp"

    def __init__(self, config: BlspConfig, new_embedding_nums):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.llama_config = LlamaConfig(**config.llama_config)
        if config.speech_encoder == 'whisper':
            self.whisper_model = WhisperEncoder(self.whisper_config)
            in_d = self.whisper_config.d_model
        elif config.speech_encoder == 'mms':
            self.whisper_model = Wav2Vec2Model.from_pretrained("/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/pretrained_models/mms-1b", _fast_init=not is_deepspeed_zero3_enabled())
            in_d = self.whisper_model.config.hidden_size

        self.llama_model = AutoModelForCausalLM.from_pretrained('/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/pretrained_models/internlm2-7b', trust_remote_code=True, _fast_init=not is_deepspeed_zero3_enabled())
        # self.llama_model = AutoModelForCausalLM.from_pretrained('/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/pretrained_models/llama2-hf-7b', trust_remote_code=True, _fast_init=not is_deepspeed_zero3_enabled())

        self.llm_type = "internlm"
        if self.llm_type == "llama":
            selected_params = ['model.embed_tokens.weight', 'lm_head.weight']
        elif self.llm_type == "internlm":
            selected_params = ['model.tok_embeddings.weight', 'output.weight']
        state_dict = self.llama_model.state_dict()
        selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_params}
        if new_embedding_nums > 0:
            if self.llm_type == "llama":
                self.llama_model.model.embed_tokens = PartiallyTrainedEmbedding(new_embedding_nums, *self.llama_model.model.embed_tokens.weight.shape, self.llama_model.model.embed_tokens.padding_idx)
                self.llama_model.lm_head = PartiallyTrainedLMHead(new_embedding_nums, *self.llama_model.model.embed_tokens.weight.shape[::-1], bias=False)
            else:
                self.llama_model.model.tok_embeddings = PartiallyTrainedEmbedding(new_embedding_nums, *self.llama_model.model.tok_embeddings.weight.shape, self.llama_model.model.tok_embeddings.padding_idx)
                self.llama_model.output = PartiallyTrainedLMHead(new_embedding_nums, *self.llama_model.model.tok_embeddings.weight.shape[::-1], bias=False)
            self.llama_model.load_state_dict(selected_state_dict, strict=False)
            self.llama_model.config.vocab_size += new_embedding_nums
            self.llama_model.vocab_size += new_embedding_nums
        out_d = self.llama_config.hidden_size
        self.subsampler = Conv1dSubsampler(
            in_d,
            2 * in_d,
            out_d,
            [int(k) for k in config.conv_kernel_sizes.split(",")],
        )
        self.use_fbank = config.use_fbank
        if self.use_fbank:
            self.fbank_downsample = Conv1dSubsampler(
                            128,
                            2 * 128,
                            1024,
                            [int(k) for k in config.conv_kernel_sizes.split(",")],
                        )
            
            self.fbank_adapter = torch.nn.Linear(1024, 4096)

        self.speech_ln = torch.nn.LayerNorm(out_d, 1e-5, True)
        self.adapter = Adapter(out_d, config.adapter_inner_dim, moe=config.moe)
    
    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.whisper_config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.whisper_config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.whisper_config.mask_time_prob,
                mask_length=self.whisper_config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.whisper_config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.whisper_config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.whisper_config.mask_feature_prob,
                mask_length=self.whisper_config.mask_feature_length,
                min_masks=self.whisper_config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features
    


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speech_values: Optional[torch.FloatTensor] = None,
        speech_attention_mask: Optional[torch.LongTensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.LongTensor] = None,
        suffix_labels: Optional[torch.LongTensor] = None,
    ):
        ### 1. forward speech
        
        speech_embeds, moe_loss, speech_attention_mask = self.get_speech_features(speech_values, speech_attention_mask)
        speech_labels = torch.LongTensor(speech_embeds.size(0), speech_embeds.size(1)).fill_(-100).to(speech_embeds.device)

        ### 2. forward llama
        if self.llm_type == "llama":
            prefix_embeds = self.llama_model.model.embed_tokens(input_ids)
            suffix_embeds = self.llama_model.model.embed_tokens(suffix_input_ids)
        else:
            prefix_embeds = self.llama_model.model.tok_embeddings(input_ids)
            suffix_embeds = self.llama_model.model.tok_embeddings(suffix_input_ids)
        
        inputs_embeds = torch.cat([prefix_embeds, speech_embeds, suffix_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, speech_attention_mask, suffix_attention_mask], dim=1)
        labels = torch.cat([labels, speech_labels, suffix_labels], dim=1)
        output = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )

        output['loss'] = output['loss'] + moe_loss
        return output
        


    def get_speech_features(self, speech_values, speech_attention_mask):
        if self.use_fbank:
            speech_values = speech_values.transpose(-2, -1)
            input_lengths = speech_attention_mask.sum(-1)
            speech_embeds, speech_lengths = self.fbank_downsample(speech_values, input_lengths)
            speech_embeds = speech_embeds.transpose(0,1)
            speech_padding_mask = lengths_to_padding_mask(speech_lengths, speech_embeds.shape[1])
            speech_atts = ~speech_padding_mask
            speech_embeds = self.fbank_adapter(speech_embeds)
            moe_loss = 0
        else:
            w2v_args = {
                "input_values": speech_values,
                "attention_mask": speech_attention_mask,
            }
            input_values = self._mask_input_features(speech_values, speech_attention_mask)
            w2v_args['input_values'] = input_values
            output = self.whisper_model(**w2v_args)
            speech_embeds = output.last_hidden_state # B x T x C
            if isinstance(self.whisper_model, WhisperEncoder):
                speech_lengths = output.output_lengths
            else:
                input_lengths = speech_attention_mask.sum(-1)
                speech_lengths = self.whisper_model._get_feat_extract_output_lengths(input_lengths)

            speech_embeds, speech_lengths = self.subsampler(speech_embeds, speech_lengths)
            speech_embeds = speech_embeds.transpose(0,1) # T x B x C -> B x T x C
            speech_padding_mask = lengths_to_padding_mask(speech_lengths)
            speech_atts = ~speech_padding_mask

            speech_embeds, moe_loss = self.adapter(speech_embeds)
        speech_embeds = self.speech_ln(speech_embeds)

        return speech_embeds, moe_loss, speech_atts

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        suffix_input_ids,
        labels=None,
        suffix_labels=None,
        attention_mask=None,
        suffix_attention_mask=None,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None,
        **kwags,
    ):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), dtype=torch.long).to(prefix_embeds.device)
        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)

        if speech_values is not None:
            speech_embeds, _, speech_attention_mask = self.get_speech_features(speech_values, speech_attention_mask)
            inputs_embeds.append(speech_embeds)
            attention_mask.append(speech_attention_mask)

        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0), suffix_embeds.size(1), dtype=torch.long).to(suffix_embeds.device)
        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwags
        )
    
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config=None
    ):
        inputs_embeds = []
        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0]
                embeds = self.llama_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0], h[1]
                speech_embeds, _, _ = self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
            else:
                raise NotImplementedError
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config
        )



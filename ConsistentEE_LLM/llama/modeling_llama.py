# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from llama.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from llama.configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

EPS_START = 0.1 #0.1 #0.3#0.06#0.05 0.01
EPS_END = 0.000#0.001#0.005
EPS_DECAY = 20000
steps_done = 0
tokens_num = 0
layers_num=0
calm_tokens_num = 0
calm_layers_num=0
cee_tokens_num = 0
cee_layers_num=0

def get_global_tokens_layers(mode=4):
    if(mode==4):
        return tokens_num, layers_num
    elif(mode==5):
        return calm_tokens_num, calm_layers_num
    elif(mode==6):
        return cee_tokens_num,cee_layers_num
    
# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        exit_recode: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        """
        if (mode == 3 or mode == 2) and past_query_key_value is not None: #即使是用我们的方法，evaluate的时候也不用这一步了

            past_query_states = past_query_key_value[0].transpose(1, 2)#.reshape(bsz * q_len, self.num_heads, self.head_dim)
            past_key_states = past_query_key_value[1].transpose(1, 2)#.reshape(bsz * q_len, self.num_heads, self.head_dim)
            past_value_states = past_query_key_value[2].transpose(1, 2)#.reshape(bsz * q_len, self.num_heads, self.head_dim)

            past_query_states = past_query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
            past_key_states = past_key_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
            past_value_states = past_value_states.reshape(bsz * q_len, self.num_heads, self.head_dim)

            #首先通过exit_recode来取对应的索引
            nonexit_indices = torch.nonzero(1-exit_recode).reshape(-1)
            hidden_states = hidden_states.reshape(bsz * q_len, -1)
            unselected_hiddenstates = hidden_states[nonexit_indices] 

            # 希望节省的时间有部分就在这里,小矩阵相乘更省时间
            new_query_states = self.q_proj(unselected_hiddenstates).view(len(nonexit_indices), self.num_heads, self.head_dim)
            new_key_states = self.k_proj(unselected_hiddenstates).view(len(nonexit_indices), self.num_heads, self.head_dim)
            new_value_states = self.v_proj(unselected_hiddenstates).view(len(nonexit_indices), self.num_heads, self.head_dim)
            #print('past_query_states.shape: ',past_query_states.shape)
            #print('past_query_states[nonexit_indices].shape: ', past_query_states[nonexit_indices].shape)
            # 把对应还没退出位置的q，k，v做插入更新
            past_query_states[nonexit_indices] = new_query_states.to(past_query_states.dtype)
            past_key_states[nonexit_indices] = new_key_states.to(past_query_states.dtype)
            past_value_states[nonexit_indices] = new_value_states.to(past_query_states.dtype)

            query_states = past_query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = past_key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = past_value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
        """

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)


        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None


        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        exit_recode: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            exit_recode=exit_recode,
            mode=mode,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)  # 这个是中间层模块的mlp，每层都有的中间mlp
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:  # 这个为false
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)


class Policy(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = gelu(self.layer1(x))
        x = gelu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        #self.Policy_networks = nn.ModuleList([Policy(config.hidden_size, 1) for _ in range(config.num_hidden_layers)])

        self.Policy_networks_0 = nn.Linear(config.hidden_size, 1)  # 因为暂时不理解module list和list为什么无法在lora中被使用，暂时先一个一个定义
        self.Policy_networks_1 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_2 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_3 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_4 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_5 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_6 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_7 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_8 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_9 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_10 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_11 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_12 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_13 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_14 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_15 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_16 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_17 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_18 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_19 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_20 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_21 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_22 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_23 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_24 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_25 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_26 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_27 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_28 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_29 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_30 = nn.Linear(config.hidden_size, 1)
        self.Policy_networks_31 = nn.Linear(config.hidden_size, 1)



        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def calculate_k_v_state(self, layer_index, hidden_states, position_ids, past_key_value, bsz, q_len):
        # 不同的layer有不同的attn, query_states可以不算
        query_states = self.layers[layer_index].self_attn.q_proj(hidden_states).view(bsz, q_len, self.layers[layer_index].self_attn.num_heads, self.layers[layer_index].self_attn.head_dim).transpose(1, 2)
        key_states = self.layers[layer_index].self_attn.k_proj(hidden_states).view(bsz, q_len,self.layers[layer_index].self_attn.num_heads,self.layers[layer_index].self_attn.head_dim).transpose(1, 2)
        value_states = self.layers[layer_index].self_attn.v_proj(hidden_states).view(bsz, q_len,self.layers[layer_index].self_attn.num_heads,self.layers[layer_index].self_attn.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.layers[layer_index].self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        return key_states, value_states


    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_layers=None,
        mode: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_logits = ()
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        bsz, q_len, _ = hidden_states.size()
        res = None

        if self.training and mode == 1:
            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]
                hidden_states_norm = self.norm(hidden_states)
                logits = output_layers[idx](hidden_states_norm)  # 每一层计算出来的logits

                all_logits += (logits,)

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        elif self.training and (mode == 2 or mode == 3): #训练时
            res = torch.randn(bsz*q_len, self.config.vocab_size).cuda() #这里的res存放的是hiddenstate
            exit_recode = torch.zeros(bsz*q_len).cuda()  # 记录已经跳出的样本, bc*seq_len
            layers = torch.ones(bsz*q_len).cuda() * self.config.num_hidden_layers  # 记录每个样本各自在哪一层退出
            probs = torch.randn(bsz*q_len).cuda()
            copy_hiddenstates = None

            for idx, decoder_layer in enumerate(self.layers):
                global steps_done  # 记录做action的次数
                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module,exit_recode, mode):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None ,exit_recode,mode)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer,exit_recode=exit_recode, mode=mode),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        exit_recode = exit_recode,
                        mode = mode,
                    )

                hidden_states = layer_outputs[0]

                if copy_hiddenstates is not None:
                    hidden_states = hidden_states.reshape(bsz * q_len, -1)
                    hidden_states[exit_recode==1] = copy_hiddenstates[exit_recode==1] #模拟copy hiddenstate的操作, 选择退出位置的hiddenstate就一直复制
                    hidden_states = hidden_states.view(bsz , q_len, -1)
                    #print('hidden_states2.shape: ', hidden_states.shape)


                copy_hiddenstates = hidden_states.reshape(bsz * q_len, -1) #记录前一层的hiddenstate
                #print('hidden_states1.shape: ',hidden_states.shape)

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                hidden_states_norm = self.norm(hidden_states)
                logits = output_layers[idx](hidden_states_norm).reshape(-1, self.config.vocab_size)

                #print('hidden_states_norm.shape: ',hidden_states_norm.shape)

                prob = torch.sigmoid(getattr(self, f"Policy_networks_{idx}")(hidden_states_norm).view(-1))

                random_tensor = torch.rand(len(prob)).cuda()
                probility = torch.randint(0, 2, size=(len(prob),)).cuda()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                math.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1


                indices = torch.nonzero((prob > 0.5) & (random_tensor > eps_threshold) & (exit_recode == 0)).reshape(-1)

                indices2 = torch.nonzero((probility == 1) & (random_tensor <= eps_threshold) & (exit_recode == 0)).reshape(-1)  # 随机性

                indices = torch.cat((indices, indices2))
                indices = torch.unique(indices)


                # print('indices: ',indices)
                # print('indices.shape: ',indices.shape)
                #相当于在这里增加一条专家数据，sample_k里面第一条，无论如何都不退出，从头算到尾， 把小于1024的，4*256（bsz*q_len），每次都过滤掉 (sample_k not defined, 后面要改为全局变量)
                # if idx <= 1: #强制要求在前25层别退，看看policy_loss
                #     indices = indices[indices >= bsz * q_len]
                # else:
                #     indices = indices[indices >= (bsz / 8) * q_len]
                indices = indices[indices >= (bsz / 4) * q_len]  #专家数据可能导致prob是1，但是却一直不退出，导致出现torch.log(prob)中有inf的问题

                exit_recode[indices] = 1  ##选择退出，不再往下算的对应下标位置赋1
                res[indices] = logits[indices].to(res.dtype)  # res只取对应退出了的一层
                layers[indices] = idx + 1  # 在第i+1层退出

                """
                probs[indices] = prob[indices].to(probs.dtype)
                """

                probs[indices] += torch.log(prob[indices] + 0.000001).to(probs.dtype)

                probs[exit_recode == 0] += torch.log(1 - prob[exit_recode == 0] + 0.000001).to(probs.dtype) #policy_gradient的公式 +0.000001 是bias，防止inf

                if 0 not in exit_recode:
                    break

                if idx == self.config.num_hidden_layers - 1:
                    res[exit_recode == 0] = logits[exit_recode == 0].to(res.dtype)

                    #probs[exit_recode == 0] = prob[exit_recode == 0].to(probs.dtype) #后面应该删去

            res = res.reshape(len(hidden_states), hidden_states.size(1), self.config.vocab_size)

        elif mode == 0 and not self.training:  ##最原始的inference
            for idx, decoder_layer in enumerate(self.layers):

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        mode=mode,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        mode=mode,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                hidden_states_norm = self.norm(hidden_states)  # 输出最后一层的hiddenstate

        #evaluate
        #original    
        elif mode==4:
            layers = self.config.num_hidden_layers - 1
            # Original_inference
            global tokens_num
            global layers_num
            if past_key_values is None:
                tokens_num = 0
                layers_num = 0
            for idx, decoder_layer in enumerate(self.layers):

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        mode=mode,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        mode = mode,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                hidden_states_norm = self.norm(hidden_states)  #在原始的inference中，这里返回的就是最后一层的hiddenstate

                if idx == self.config.num_hidden_layers - 1:
                    layers=idx
                    # print('original_idx: ',idx)
                    res = output_layers[idx](hidden_states_norm)
                    break
            tokens_num +=1
            layers_num+=layers
            # print('original_avg_layers: ', layers_num/tokens_num)

        elif mode==5:# CALM
            layers = self.config.num_hidden_layers - 1
            # CALM
            previous_hiddenstates = None
            #print('hidden_states.shape: ',hidden_states.shape)
            global calm_tokens_num
            global calm_layers_num
            if past_key_values is None:
                calm_tokens_num = 0
                calm_layers_num = 0
                
            for idx, decoder_layer in enumerate(self.layers):

                past_key_value = past_key_values[idx] if past_key_values is not None else None
                # if past_key_values is not None:
                #     # print(idx,' len(past_key_values): ', len(past_key_values))
                #     # print(idx, ' past_key_values[idx][0].shape: ', past_key_values[idx][0].shape)

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        mode=mode,
                    )
                else:
                    #print('use_cache: ', use_cache)

                    layer_outputs = decoder_layer( #evaluate的时候走这边
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        mode=mode,
                    )

                hidden_states = layer_outputs[0]

                hidden_states_norm = self.norm(hidden_states)
                # print('hidden_states_norm.shape: ', hidden_states_norm.shape)
                similarity = 0
                if past_key_values is not None and previous_hiddenstates is not None:
                    similarity = torch.cosine_similarity(hidden_states_norm, previous_hiddenstates, dim=-1)

                previous_hiddenstates = hidden_states_norm


                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if past_key_values is not None:
                    if similarity>0.975 or idx == self.config.num_hidden_layers - 1: #选择退出

                        layers = idx
                        # print('calm_layers: ', layers)

                        for iy in range(idx + 1, self.config.num_hidden_layers):
                            # 就是对应退出的某一层的k、vstates (kstates,vstates)
                            #print('past_key_values[iy][0].shape: ', past_key_values[iy][0].shape)

                            # print('layer_outputs[2 if output_attentions else 1][0].shape: ',layer_outputs[2 if output_attentions else 1][0].shape)
                            # print('layer_outputs[2 if output_attentions else 1][0][:, :, -1:, :].shape: ',layer_outputs[2 if output_attentions else 1][0][:, :, -1:, :].shape)

                            key_states, value_states = self.calculate_k_v_state(layer_index = iy, hidden_states = hidden_states, position_ids = position_ids, past_key_value = past_key_value, bsz=bsz, q_len=q_len)

                            # past_key_states = torch.cat((past_key_values[iy][0], layer_outputs[2 if output_attentions else 1][0][:, :, -1:, :]),dim=2)  #layer_outputs[2 if output_attentions else 1][0][:, :, -1:, :])指的是句子中的最后一个token
                            # past_values_states = torch.cat((past_key_values[iy][1], layer_outputs[2 if output_attentions else 1][1][:, :, -1:, :]),dim=2)

                            past_key_states = torch.cat((past_key_values[iy][0], key_states),dim=2)
                            past_values_states = torch.cat((past_key_values[iy][1], value_states),dim=2)
                            # print('past_key_states.shape: ',past_key_states.shape)
                            next_decoder_cache += ((past_key_states,past_values_states),)

                        break  # 就假如break的话处理一下cache，非常简单的事情
            calm_tokens_num +=1
            calm_layers_num+=layers
            print('CALM_avg_layers: ', calm_layers_num/calm_tokens_num)
            res = output_layers[layers](hidden_states_norm)

        #ConsistentEE
        elif mode==6:#ConsistentEE
            layers = self.config.num_hidden_layers - 1
            # for test
            global cee_tokens_num
            global cee_layers_num
            if past_key_values is None:
                cee_tokens_num = 0
                cee_layers_num = 0

             # ConsistentEE
            for idx, decoder_layer in enumerate(self.layers):

                past_key_value = past_key_values[idx] if past_key_values is not None else None
                # if past_key_values is not None:
                #     print(idx,' len(past_key_values): ', len(past_key_values))
                #     print(idx, ' past_key_values[idx][0].shape: ', past_key_values[idx][0].shape)

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        mode=mode,
                    )
                else:
                    #print('use_cache: ', use_cache)

                    layer_outputs = decoder_layer( #evaluate的时候走这边
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        mode=mode,
                    )

                hidden_states = layer_outputs[0]

                hidden_states_norm = self.norm(hidden_states)
                # print('hidden_states_norm.shape: ', hidden_states_norm.shape)

                if past_key_values is not None: #如果非生成第一个token的时候
                    # print('self.Policy_networks[idx](hidden_states_norm).shape: ',self.Policy_networks[idx](hidden_states_norm).shape)
                    prob = torch.sigmoid(getattr(self, f"Policy_networks_{idx}")(hidden_states_norm))[0, 0, 0]

                    #print('prob: ', prob)


                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if past_key_values is not None:
                    if  prob > 0.5 or idx == self.config.num_hidden_layers - 1: #决定退出或者到达最后一层 (tokens_num >=5 and

                        layers = idx
                        # print('con_layers: ', layers)

                        for iy in range(idx + 1, self.config.num_hidden_layers):
                            # 就是对应退出的某一层的k、vstates (kstates,vstates)
                            # print('past_key_values[iy][0].shape: ', past_key_values[iy][0].shape)
                            # print('layer_outputs[2 if output_attentions else 1][0].shape: ',layer_outputs[2 if output_attentions else 1][0].shape)
                            # print('layer_outputs[2 if output_attentions else 1][0][:, :, -1:, :].shape: ',layer_outputs[2 if output_attentions else 1][0][:, :, -1:, :].shape)

                            key_states, value_states = self.calculate_k_v_state(layer_index=iy, hidden_states=hidden_states, position_ids=position_ids, past_key_value=past_key_value, bsz=bsz, q_len=q_len)

                            past_key_states = torch.cat((past_key_values[iy][0], key_states), dim=2)
                            past_values_states = torch.cat((past_key_values[iy][1], value_states), dim=2)

                            next_decoder_cache += ((past_key_states,past_values_states),)

                        break  # 就假如break的话处理一下cache，非常简单的事情

            res = output_layers[layers](hidden_states_norm)

            cee_tokens_num +=1
            cee_layers_num+=layers
            print('con_avg_layers: ', cee_layers_num/cee_tokens_num)
                #
                # print('next_decoder_cache[0][0].shape: ',next_decoder_cache[0][0].shape)
                #print('len(next_decoder_cache): ',len(next_decoder_cache)) #指的是有32层




        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states_norm, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states_norm,
            past_key_values=next_cache,
            all_logits=all_logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            chosen_logits=res if ((mode == 2 or mode == 3) or not self.training) else None,
            layers=layers if ((mode == 2 or mode == 3) or not self.training) else None,
            probs=probs if (mode == 2 or mode == 3) else None,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_heads_0 = nn.Linear(config.hidden_size, config.vocab_size,
                                    bias=False)  # 因为暂时不理解module list和list为什么无法在lora中被使用，暂时先一个一个定义
        self.lm_heads_1 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_2 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_3 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_4 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_5 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_6 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_7 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_8 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_9 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_10 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_11 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_12 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_13 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_14 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_15 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_16 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_17 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_18 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_19 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_20 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_21 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_22 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_23 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_24 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_25 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_26 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_27 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_28 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_29 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_30 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_heads_31 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
        mode: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_layers=[getattr(self, f"lm_heads_{i}") for i in range(self.config.num_hidden_layers)],
            mode=mode,#推理时直接指定，训练时是mode,
        )

        hidden_states = outputs[0]
        loss = None

        if mode == 1 and self.training:
            # logits = self.lm_heads_30(hidden_states)
            logits = outputs.all_logits[-1]

            all_logits = outputs.all_logits

            loss = None
            if labels is not None:

                """
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                """
                total_loss = None
                total_weights = 0
                loss_fct = CrossEntropyLoss()

                shift_labels = labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)

                for ix, logits_item in enumerate(all_logits):
                    shift_logits = logits_item[..., :-1, :].contiguous()

                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)

                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss += loss * (ix + 1)
                    total_weights += ix + 1
                loss = total_loss / total_weights

        elif (mode == 2 or mode == 3) and self.training:

            logits = outputs.chosen_logits
            # print('logits.shape:', logits.shape)
            # print('labels.shape:', labels.shape)
            layers = outputs.layers.view(-1)
            probs = outputs.probs.view(-1)


            probs = probs/ layers

            # frequency = [0.004971716712680805, 0.0054688883839488864, 0.006015777222343776, 0.006617354944578153, 0.007279090439035969, 0.008006999482939568, 0.008807699431233525, 0.009688469374356879, 0.010657316311792567, 0.011723047942971824, 0.012895352737269007, 0.014184888010995909, 0.015603376812095502, 0.017163714493305053, 0.01888008594263556, 0.020768094536899116, 0.022844903990589034, 0.025129394389647938, 0.027642333828612732, 0.030406567211474007, 0.03344722393262141, 0.036791946325883555, 0.040471140958471916, 0.044518255054319104, 0.04897008055975103, 0.05386708861572613, 0.05925379747729875, 0.06517917722502863, 0.07169709494753149, 0.07886680444228465, 0.08675348488651313, 0.09542883337516445]
            frequency = [0.0017328005712952785, 0.0019927206569895703, 0.002291628755538006, 0.002635373068868706,
                         0.003030679029199012, 0.0034852808835788636, 0.004008073016115693, 0.004609283968533046,
                         0.005300676563813003, 0.006095778048384952, 0.0070101447556426945, 0.008061666468989098,
                         0.009270916439337461, 0.010661553905238081, 0.012260786991023792, 0.014099905039677358,
                         0.016214890795628965, 0.018647124414973303, 0.021444193077219297, 0.024660822038802193,
                         0.02835994534462252, 0.03261393714631589, 0.03750602771826327, 0.04313193187600276,
                         0.049601721657403175, 0.05704197990601365, 0.06559827689191569, 0.07543801842570304,
                         0.08675372118955849, 0.09976677936799225, 0.11473179627319108, 0.13194156571416973]

            print('avg_layers: ', torch.sum(layers) / len(layers))

            if labels is not None:
                # Shift so that tokens < n predict n

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)

                # print('shift_logits.shape:', shift_logits.shape)
                # print('shift_labels.shape:', shift_labels.shape)
                loss = loss_fct(shift_logits, shift_labels)  # 所以需要看一下这里的loss的形态

            # print('loss: ',loss)
            # print('loss.shape: ', loss.shape)
            L = len(loss)
            # print('L:',L)
            classfiy_loss = sum(loss) / L
            #print('classfiy_loss: ',classfiy_loss)

            sample_K = 4
            bsz = L // sample_K  # sample_K

            rewards = [[] for _ in range(bsz)]

            for i in range(L):
                reward = -loss[i].item() - layers[i] * alpha  # + frequency[int(layers[i])-1] * 15 #- layers[i] * 0.010#* (1 - batch[5][i] / 12)
                # print(f"loss:{loss[i].item()}")
                # print(f"alpha :{alpha}")
                # print('loss[i].item(): ',loss[i].item())
                # print('layers[i]: ',layers[i])
                rewards[i % bsz].append(reward)

            policy_loss = 0

            baseline_sums = [sum(rs) for rs in rewards]
            baseline_lens = [len(rs) for rs in rewards]
            # print('probs: ',probs)
            # print('rewards: ',rewards)

            for i, rs in enumerate(rewards):
                baseline = baseline_sums[i] / baseline_lens[i]
                for j in range(sample_K):
                    reward = (rs[j] - baseline)
                    # print('reward: ',reward)
                    policy_loss += reward * probs[j * bsz + i] * -1
                # print('xxxxxxxxxxxxxx')

            policy_loss = policy_loss / L
            # print('policy_loss_mean: ', policy_loss)
            # print('classfiy_loss_mean: ', classfiy_loss)
            #print('torch.mean(probs): ', torch.mean(probs))



            if mode == 2:
                loss = policy_loss
            elif mode == 3:
                loss = classfiy_loss  #+ classfiy_loss
            print(' loss',loss)

        else: # 先暂时统一都用这个原先的reference

            logits = outputs.chosen_logits

            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)



        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss if loss is not None else None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter

# TODO: Add in attention mask to the entire assignment

# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        # d_model is hidden embedding lengths 
        self.n_heads = config.n_heads
        # self.vocab_size = config.vocab_size

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.attn_dropout = nn.Dropout(p=0.1)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask # keep only the lower triangle 
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).bool().view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        
        batch_size, seq_len, d_model = x.shape
        n_heads = self.n_heads
        d_attention = self.d_attention
        
        big_Q = self.W_q(x)
        big_K = self.W_k(x)
        big_V = self.W_v(x)
        
        # new shape is (batch_size, n_heads, seq_len, d_attention)
        Q = big_Q.reshape(batch_size, seq_len, n_heads, d_attention).transpose(1, 2)
        K = big_K.reshape(batch_size, seq_len, n_heads, d_attention).transpose(1, 2)
        V = big_V.reshape(batch_size, seq_len, n_heads, d_attention).transpose(1, 2)


        # it has the size of (batch_size, n_heads, seq_len, seq_len) seq_len = length of tokens
        raw_attention = Q @ K.transpose(2, 3) / math.sqrt(d_attention)

        masked_attention = raw_attention.masked_fill(~self.causal_mask[:,:,:seq_len,:seq_len], float("-inf"))

        after_softmax = softmax(masked_attention, dim = 3)
        after_softmax = self.attn_dropout(after_softmax)

        H = after_softmax @ V # (batch_size, n_heads, seq_len, d_attention)

        H = H.transpose(1, 2).reshape(batch_size, seq_len, d_model) 

        return self.W_o(H)  # (batch_size, seq_len, d_model)


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO complete
        new_x = x + self.attention(self.pre_layer_norm(x))
        x = new_x + self.mlp(self.post_layer_norm(new_x))

        return x


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        # TODO, complete
        batch_size, seq_len = x.shape
        token_embedding = self.embeddings(x) #(batch_size, seq_len, d_model)
        positions = torch.arange(seq_len, device = x.device)
        position_embedding = self.position_embeddings(positions).unsqueeze(0)
        final_embedding = token_embedding + position_embedding  #(batch_size, seq_len, d_model)

        for building_block in self.backbone:
            final_embedding = building_block(final_embedding)
        
        result = self.lm_head(self.final_layer_norm(final_embedding))

        return result

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:
    
        for _ in range(num_new_tokens):
            if x.shape[1] > self.config.context_length:
                new_x = x[:, -self.config.context_length:]
            else:
                new_x = x

            forward_output = self.forward(new_x) # (batch, seq_len, vocab_size)
            #the next token is the logit output of the current last word
            last_forward_output  = forward_output[:,-1,:]
            next_token = torch.argmax(last_forward_output, dim = -1, keepdim = True)
            # print(x.shape)
            # print(next_token.shape)
            x = torch.concat([x, next_token], dim = 1)

        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"], 
    ) -> Float[Tensor, ""]:
        
        #so we are using the first 0:seq_len-1 to predict 1:to seq_len

        batch_size, seq_len = input_ids.shape

        y_hat = self.forward(input_ids[:,:seq_len-1]).reshape(-1,self.vocab_size) #(batch, seq_len-1, vocab_size)

        y = input_ids[:,1:].reshape(-1)

        return F.cross_entropy(y_hat, y)


    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # Uncomment this if you are not logged in
    # huggingface_hub.login()
    
    model = Transformer.from_pretrained()

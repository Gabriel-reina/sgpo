"""Randomly initialized history encoder + causal SID decoder."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    levels: int = 3
    code_size: int = 256
    max_history: int = 10
    hidden_size: int = 128
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    ffn_size: int = 384
    dropout: float = .1


class GenerativeRecommender(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.code_embeddings = nn.ModuleList([nn.Embedding(config.code_size + 1, d,
            padding_idx=config.code_size) for _ in range(config.levels)])
        self.item_projection = nn.Linear(config.levels * d, d)
        self.history_positions = nn.Embedding(config.max_history, d)
        self.sid_positions = nn.Embedding(config.levels, d)
        self.bos = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.empty_history = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, config.heads,
            config.ffn_size, config.dropout, batch_first=True, norm_first=True),
            config.encoder_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, config.heads,
            config.ffn_size, config.dropout, batch_first=True, norm_first=True),
            config.decoder_layers, norm=nn.LayerNorm(d))
        self.output_heads = nn.ModuleList([nn.Linear(d, config.code_size) for _ in range(config.levels)])

    def encode(self, history):
        padding = history[:, :, 0] == self.config.code_size
        h = self.item_projection(torch.cat([emb(history[:, :, i]) for i, emb in enumerate(self.code_embeddings)], -1))
        h = h + self.history_positions(torch.arange(history.shape[1], device=history.device))[None]
        # Empty histories still need one real memory position to avoid all-masked attention.
        empty = padding.all(dim=1)
        h = h.clone()
        padding = padding.clone()
        h[empty, 0] = self.empty_history
        padding[empty, 0] = False
        memory = self.encoder(h, src_key_padding_mask=padding)
        return memory, padding

    def decode(self, memory, padding, prefix):
        # [BOS, previous SID tokens]; current/future targets never enter their own prediction.
        batch, depth = prefix.shape
        steps = [self.bos.expand(batch, -1)]
        steps += [self.code_embeddings[i](prefix[:, i]) for i in range(depth)]
        h = torch.stack(steps, 1)
        h = h + self.sid_positions(torch.arange(depth + 1, device=prefix.device))[None]
        causal = torch.triu(torch.ones(depth + 1, depth + 1, device=prefix.device, dtype=torch.bool), diagonal=1)
        h = self.decoder(h, memory, tgt_mask=causal, memory_key_padding_mask=padding)
        return torch.stack([self.output_heads[i](h[:, i]) for i in range(depth + 1)], 1)

    def forward(self, history, target):
        memory, padding = self.encode(history)
        return self.decode(memory, padding, target[:, :-1])

    def checkpoint_config(self):
        return asdict(self.config)

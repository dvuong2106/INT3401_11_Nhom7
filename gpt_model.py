"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
from models import Attention


class Transformer_Block(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()

        self.attn_block = Attention(n_embd, block_size)
        self.norm_1 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        """YOUR CODE HERE"""
        x = self.norm_1(x + self.attn_block(x))
        x = self.norm_2(x + F.relu(self.linear_1(x)))
        return x


class Character_GPT(nn.Module):

    def __init__(self, block_size, n_embd, n_layer, vocab_size):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)


        self.transformer_blocks = nn.ModuleList([Transformer_Block(n_embd, block_size) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, vocab_size, bias=False)


    def get_loss(self, input, target):
        output = self(input)
        return F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=-1)

    def forward(self, input):
        b, t = input.size()
        assert t <= self.block_size, f"Không thể forward chuỗi có độ dài {t}, kích thước khối (block size) chỉ là {self.block_size}"

        """YOUR CODE HERE"""
        x = self.embed(input)


        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.output_layer(x)


        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]


            probs = F.softmax(logits, dim=-1)


            idx_next = torch.multinomial(probs, num_samples=1)


            idx = torch.cat((idx, idx_next), dim=1)

        return idx

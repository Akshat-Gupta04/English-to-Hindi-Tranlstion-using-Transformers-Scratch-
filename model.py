import torch
import math
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x => (batch_size, seq_len)
        return self.embedding(x.long()) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings for max seq_len
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape => (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x => (batch_size, seq_len, d_model)
        seq_len = x.size(1)            
        # Add positional encoding up to current seq_len
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)

class layerNormalization(nn.Module):
    def __init__(self, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x => (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (B, seq_len, d_model) => (B, seq_len, d_ff) => (B, seq_len, d_model)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi‐head attention that can handle cross‐attention
    where query has shape [B, T_q, d_model] and
    key/value have shape [B, T_k, d_model] with T_q != T_k.
    """
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        """
        query => (B, h, T_q, d_k)
        key   => (B, h, T_k, d_k)
        value => (B, h, T_k, d_k)
        mask  => can be shape (B, 1, T_k) or (B, T_q, T_k) or broadcastable
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # if mask shape is (1, T_q, T_k) or (B, T_q, T_k), that can broadcast
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)

        out = torch.matmul(attn, value)
        return out, attn  # (B, h, T_q, d_k), (B, h, T_q, T_k)

    def forward(self, q, k, v, mask):
        # q => (B, T_q, d_model)
        # k => (B, T_k, d_model)
        # v => (B, T_k, d_model)
        B_q, len_q, _ = q.shape
        B_k, len_k, _ = k.shape
        B_v, len_v, _ = v.shape

        # 1) Project to multi‐head
        query = self.w_q(q)  # [B_q, T_q, d_model]
        key   = self.w_k(k)  # [B_k, T_k, d_model]
        value = self.w_v(v)  # [B_v, T_k, d_model]

        # 2) Reshape => [B, h, seq_len, d_k]
        query = query.view(B_q, len_q, self.h, self.d_k).transpose(1, 2)  # (B_q, h, T_q, d_k)
        key   = key.view(B_k, len_k, self.h, self.d_k).transpose(1, 2)   # (B_k, h, T_k, d_k)
        value = value.view(B_v, len_k, self.h, self.d_k).transpose(1, 2) # (B_v, h, T_k, d_k)

        # 3) Apply scaled dot‐product attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        # x => (B_q, h, T_q, d_k)

        # 4) Reshape back
        x = x.transpose(1, 2).contiguous()  # => (B_q, T_q, h, d_k)
        x = x.view(B_q, len_q, self.d_model) # => (B_q, T_q, d_model)

        return self.w_o(x)  # => (B_q, T_q, d_model)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layerNormalization()

    def forward(self, x, sublayer):
        """
        x => (B, seq_len, d_model)
        sublayer => a function that takes in x and returns x'
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        # 1) Self‐Attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # 2) Feed Forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, src_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.src_attention_block = src_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1) Self‐Attention
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # 2) Cross‐Attention
        x = self.residual_connections[1](
            x, lambda x: self.src_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        # 3) Feed‐Forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x => (B, seq_len, d_model)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(
        self,
        encoder:Encoder,
        decoder:Decoder,
        src_embed:InputEmbedding,
        tgt_embed:InputEmbedding,
        src_pos:PositionalEncoding,
        tgt_pos:PositionalEncoding,
        projection_layer:ProjectionLayer
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        src => (B, seq_len)
        src_mask => (B, 1, seq_len) or broadcastable
        """
        src = self.src_embed(src)  # => (B, seq_len, d_model)
        src = self.src_pos(src)    # => (B, seq_len, d_model)
        return self.encoder(src, src_mask)  # => (B, seq_len, d_model)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        tgt => (B, seq_len)
        encoder_output => (B, src_len, d_model)
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # x => (B, seq_len, d_model)
        return self.projection_layer(x)

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:

    # Embeddings & position
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Encoder
    encoder_blocks = []
    for _ in range(N):
        enc_self_attention = MultiHeadAttention(d_model, h, dropout)
        enc_ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(enc_self_attention, enc_ff, dropout)
        encoder_blocks.append(encoder_layer)

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Decoder
    decoder_blocks = []
    for _ in range(N):
        dec_self_attention = MultiHeadAttention(d_model, h, dropout)
        dec_cross_attention = MultiHeadAttention(d_model, h, dropout)
        dec_ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_layer = DecoderBlock(dec_self_attention, dec_cross_attention, dec_ff, dropout)
        decoder_blocks.append(decoder_layer)

    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        projection_layer
    )

    # Initialize parameters with Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
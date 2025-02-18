import torch
from torch.utils.data import Dataset
from config import get_config

class BilingualDataset(Dataset):
    """
    A simple dataset that takes a HuggingFace split (ds),
    tokenizes each sentence pair, and returns:
      - encoder_input
      - decoder_input
      - encoder_mask
      - decoder_mask
      - label
      - original src_text, tgt_text
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        # or self.seq_len = seq_len  (both approaches are fine)
        self.seq_len = get_config()["seq_len"]
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.long)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Tokenize sentences
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate if too long
        enc_input_tokens = enc_input_tokens[: self.seq_len - 2]  # Reserve space for [SOS] + [EOS]
        dec_input_tokens = dec_input_tokens[: self.seq_len - 1]  # Reserve space for [SOS]

        # Calculate padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Encoder input: [SOS] + sentence + [EOS] + [PAD]...
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.long),
        ], dim=0)

        # Decoder input: [SOS] + sentence + [PAD]...
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.long),
            torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.long),
        ], dim=0)

        # Target label: sentence + [EOS] + [PAD]...
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.long),
        ], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # shape => (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len),
            # shape => (1, seq_len) & (1, seq_len, seq_len) => (1, seq_len, seq_len) after broadcast
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    """Creates a causal (future) mask for the decoder, shape: (1, size, size)."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
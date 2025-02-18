from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

# Hugging Face Datasets & Tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """Performs greedy decoding for inference."""
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Encode the source
    encoder_output = model.encode(source, source_mask)
    # Decoder starts with <SOS>
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    while decoder_input.size(1) < max_len:
        # Build a causal + padding mask for the current decoder_input
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        # (tgt, encoder_output, src_mask, tgt_mask)
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # Last token's distribution
        prob = model.project(out[:, -1])  # shape: (1, vocab_size)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_all_sentences(ds, lang):
    """Yield all sentences for a given language. Used for tokenizer training."""
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    """Load or train a tokenizer for a language (en or hi)."""
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to: {tokenizer_path}")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Tokenizer loaded from: {tokenizer_path}")
    return tokenizer

def get_ds(config):
    """
    Load the IITB English-Hindi dataset from Hugging Face,
    but use the 'validation' split for training, 'test' split for validation/test.
    """
    dataset = load_dataset("cfilt/iitb-english-hindi")

    # train: validation split
    train_ds_raw = dataset["validation"]
    # val/test: test split
    test_ds_raw = dataset["test"]

    # ---- Print a small preview of the chosen training data ----
    df_train_preview = pd.DataFrame({
        "english_text": [train_ds_raw[i]["translation"]["en"] for i in range(5)],
        "hindi_text":   [train_ds_raw[i]["translation"]["hi"] for i in range(5)]
    })
    print(f"\n[INFO] Training dataset size (using 'validation' split): {len(train_ds_raw)}")
    print("[INFO] Preview of first 5 examples (English → Hindi):\n")
    print(df_train_preview)
    print("-------------------------------------------------------\n")

    # Build or load tokenizers from the training data
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config["lang_tgt"])

    # Print vocabulary details
    vocab_src = tokenizer_src.get_vocab()  # dict: token -> ID
    vocab_tgt = tokenizer_tgt.get_vocab()  # dict: token -> ID

    print(f"[INFO] Source vocab size: {tokenizer_src.get_vocab_size()}")
    src_vocab_items = list(vocab_src.items())[:20]  # first 20
    print("[INFO] Sample (token, id) from source vocab:\n", src_vocab_items)
    print("-------------------------------------------------------")
    print(f"[INFO] Target vocab size: {tokenizer_tgt.get_vocab_size()}")
    tgt_vocab_items = list(vocab_tgt.items())[:20]  # first 20
    print("[INFO] Sample (token, id) from target vocab:\n", tgt_vocab_items)
    print("=======================================================\n")

    # Build PyTorch dataset objects
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )
    test_ds = BilingualDataset(
        test_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=4,  # bigger batch for validation speed
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """Initialize the Transformer model."""
    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"]
    )
    return model

def run_validation(model, val_dataloader, device, loss_fn, tokenizer_src, tokenizer_tgt):
    """
    Run a validation pass on val_dataloader:
    - Returns the average loss,
    - Prints a few (source, target, predicted) examples at the end.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        # Accumulate val loss
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device).long()
            decoder_input = batch["decoder_input"].to(device).long()
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device).long()

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()
            total_count += 1

    # Average validation loss
    avg_loss = total_loss / max(total_count, 1)

    # --- Show a few predictions from a small sample batch ---
    try:
        sample_batch = next(iter(val_dataloader))
    except StopIteration:
        print("[WARNING] Validation dataloader is empty; cannot show predictions.")
        return avg_loss

    max_examples = min(3, sample_batch["encoder_input"].size(0))
    with torch.no_grad():
        for idx in range(max_examples):
            src_seq = sample_batch["encoder_input"][idx].unsqueeze(0).to(device).long()
            src_mask = sample_batch["encoder_mask"][idx].unsqueeze(0).to(device)

            src_text_str = sample_batch["src_text"][idx]
            tgt_text_str = sample_batch["tgt_text"][idx]

            # Use greedy_decode
            predicted_ids = greedy_decode(
                model,
                src_seq,
                src_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len=50,
                device=device
            )
            predicted_text = tokenizer_tgt.decode(predicted_ids.cpu().numpy().tolist())

            print(f"\n[VAL] Example {idx+1}:")
            print(f"  Source:    {src_text_str}")
            print(f"  Target:    {tgt_text_str}")
            print(f"  Predicted: {predicted_text}")

    return avg_loss

def train_model(config):
    """Train the Transformer model using the 'validation' dataset for training, and test set for validation."""
    print(f"Training on: {device}")

    # Create the weights folder if it doesn't exist
    weights_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    Path(weights_folder).mkdir(parents=True, exist_ok=True)

    # Dataloaders and tokenizers
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Build model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # CrossEntropyLoss must ignore the target PAD token
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1
    ).to(device)

    initial_epoch = 0
    global_step = 0

    # Possibly resume from a previous checkpoint
    model_filename = None
    if config["preload"] == "latest":
        model_filename = latest_weights_file_path(config)
    elif config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])

    if model_filename and Path(model_filename).exists():
        print(f"[INFO] Resuming training from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print("[INFO] Training from scratch.")

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device).long()
            decoder_input = batch["decoder_input"].to(device).long()
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device).long()

            # Encode
            encoder_output = model.encode(encoder_input, encoder_mask)
            # Decode
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            # Project => get logits
            proj_output = model.project(decoder_output)

            # Compute loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})
            writer.add_scalar("Train Loss", loss.item(), global_step)
            writer.flush()

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # After each epoch, run validation
        val_loss = run_validation(
            model,
            test_dataloader,
            device,
            loss_fn,
            tokenizer_src,
            tokenizer_tgt
        )
        print(f"\n[INFO] Epoch {epoch} Validation Loss: {val_loss:.4f}")
        writer.add_scalar("Val Loss", val_loss, epoch)

        # Save checkpoint
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
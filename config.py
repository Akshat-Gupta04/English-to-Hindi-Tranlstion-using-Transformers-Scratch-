from pathlib import Path

def get_config():
    return {
        "batch_size": 4,  
        "num_epochs": 2,
        "lr": 1e-4,
        "seq_len": 256,  
        "d_model": 256,
        "datasource": "En-hi translation",  # The base dataset
        "lang_src": "en",  # Source: English
        "lang_tgt": "hi",  # Target: Hindi
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",  # can be None or 'latest' or a string like '01'
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_en_hi"  
    }

def get_weights_file_path(config, epoch: str):
    """Return the path to the model weights file for a given epoch."""
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    """Find the latest weights file in the weights folder."""
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])
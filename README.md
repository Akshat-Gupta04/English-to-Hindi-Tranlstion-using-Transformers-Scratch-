# English to Hindi Translation using Transformer from scratch
 

## Overview
This repository contains a full implementation of a **Transformer-based Neural Machine Translation (NMT) model** trained from scratch for **English-to-Hindi translation**. The model is implemented using **PyTorch** and follows the architecture outlined in the **"Attention Is All You Need"** paper.

## Features
✅ Implements **Self-Attention, Multi-Head Attention, and Positional Encoding** from scratch.
✅ Uses **custom tokenizers** trained on the dataset.
✅ Supports **CUDA acceleration** for faster training.
✅ Evaluates using **greedy decoding** for predictions.
✅ Provides **training & validation accuracy/loss tracking**.

---

## Training Details
🖥️ **Training Device:** CUDA
📊 **Train Dataset Size:** 1,659,083
📊 **Validation Dataset Size:** 520
⏳ **Total Training Time:** **74,026.96 seconds (~20.56 hours)**
📈 **Total Epochs:** 10

### Epoch Training Times:
```
Epoch 0 took 7221.75 seconds.
Epoch 1 took 7276.88 seconds.
Epoch 2 took 7248.24 seconds.
Epoch 3 took 7285.44 seconds.
Epoch 4 took 7276.41 seconds.
Epoch 5 took 7163.37 seconds.
Epoch 6 took 7117.83 seconds.
Epoch 7 took 7223.12 seconds.
Epoch 8 took 7441.33 seconds.
Epoch 9 took 7971.59 seconds.
```

---

## Model Architecture
The model follows the **Transformer** architecture with the following components:
- **Input Embedding Layer**: Converts token indices to dense vectors.
- **Positional Encoding**: Adds positional information to embeddings.
- **Encoder & Decoder Layers**: Stacks of self-attention and feed-forward layers.
- **Multi-Head Attention Mechanisms**: Improves contextual understanding.
- **Projection Layer**: Outputs probabilities over the vocabulary.

---



## Usage
### Training
To train the model, run:
```bash
$ python train.py
```

### Sample Prediction
| Source | Expected | Predicted |
|--------|---------|-----------|
| NHPS officials and staff were working hard to reinstate the power supply. | एनएचपीसी के अधिकारी व कर्मचारी तीन दिन से लगातार बिजली सेवा बहाल करने में लगे हुए थे। | बिजली आपूर्ति को रोकने के लिए अधिकारियों और कर्मचारियों को कठोर परिश्रम करना पड़ता था। |

---

## Next Steps
🔹 Improve translation accuracy using **beam search decoding**.
🔹 Fine-tune model on **larger, domain-specific datasets**.
🔹 Optimize tokenization with **subword embeddings (BPE)**.


Would love feedback and suggestions! 🚀🔥

#DeepLearning #NLP #Transformers #MachineTranslation #AI

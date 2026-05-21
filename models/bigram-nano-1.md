---
license: apache-2.0
language:
  - vi
library_name: safetensors
tags:
  - vietnamese
  - conversational
  - custom-architecture
  - bigram
  - aevynt-lab
pipeline_tag: text-generation
---

# Bigram Nano 1

Bigram Nano 1 is a compact Vietnamese conversational model developed by Aevynt Lab using the custom Bigram recurrent-depth architecture. It is intended for lightweight local experimentation, identity and greeting behavior, short Vietnamese chat responses, and small-scale research around tone-aware tokenization.

This repository contains portable `safetensors` weights plus the tokenizer and model configuration needed to load the checkpoint with the Bigram codebase.

## Model Details

- **Model name:** Bigram Nano 1
- **Developer:** Aevynt Lab
- **Language:** Vietnamese
- **Architecture:** Bigram recurrent-depth transformer
- **Parameters:** 1,174,657
- **Sequence length:** 128
- **Tokenizer:** Bigram tonal tokenizer
- **Checkpoint source:** `nano1/sft/ckpt_final.pt`
- **Weights format:** `model.safetensors`
- **License:** Apache-2.0

## Intended Use

Bigram Nano 1 is suitable for:

- Vietnamese greeting and identity demos
- Small local inference experiments
- Testing the Bigram tokenizer and architecture
- Educational examples of compact custom language models

It is not intended for production decision-making, medical, legal, financial, or safety-critical use.

## Example Prompts

```text
xin chào!
```

```text
bạn là ai?
```

```text
mày ăn cơm chưa?
```

```text
giá vàng hôm nay bao nhiêu?
```

```text
tạm biệt
```

## Loading Example

Clone the Bigram codebase first:

```bash
git clone https://github.com/aevynt/bigram.git
cd bigram
pip install -r requirements.txt
pip install safetensors huggingface_hub
```

Download this model repository and load it with the custom Bigram architecture:

```python
import json
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from bigram import BigramModel, BigramTokenizer
from bigram.config import ModelConfig

model_dir = snapshot_download("aevynt/bigram-nano-1")

with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

model_cfg = ModelConfig(**cfg["model"])
model = BigramModel(model_cfg)
model.load_state_dict(load_file(f"{model_dir}/model.safetensors"))
model.eval()

tokenizer = BigramTokenizer.load(f"{model_dir}/tokenizer.json")
```

You can also use `sample_inference.py` from the model repository as a minimal local inference example.

## Limitations

- The model is very small and may produce incorrect, repetitive, or incomplete answers.
- It does not have access to real-time information.
- It may refuse or answer vaguely for prompts outside its narrow training distribution.
- It uses a custom architecture, so it is not directly loadable with `AutoModelForCausalLM`.

## Training Data

The model was trained on small Vietnamese instruction and conversational datasets prepared in the Bigram repository, including identity, greeting, out-of-scope, and short dialogue examples.

## Citation

```bibtex
@misc{aevynt_bigram_nano_1_2026,
  title = {Bigram Nano 1},
  author = {Aevynt Lab},
  year = {2026},
  url = {https://huggingface.co/aevynt/bigram-nano-1}
}
```

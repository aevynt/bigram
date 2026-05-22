# Bigram

Bigram is a compact Vietnamese-first language model stack built around recurrent-depth transformers. The project is meant for training small models from scratch, experimenting with test-time compute, and building tool-routing assistants without depending on a pretrained base model.

Published model:

- Hugging Face: https://huggingface.co/aevynt/bigram-nano-1
- Local model card: [models/bigram-nano-1.md](models/bigram-nano-1.md)

## What Changed

The current codebase includes an upgraded Bigram core:

- Deterministic recurrent state initialization by default, so evaluation and generation are stable for a fixed prompt and sampling seed.
- Optional adaptive recurrent early-exit during evaluation, which can stop extra recurrent steps when the latent state has converged.
- Improved generation controls with `top_p` nucleus sampling and `repetition_penalty`.
- The original stochastic latent initialization remains available through `state_init_mode="normal"` for experiments.

## Architecture

Bigram uses a prelude/recurrent/coda layout:

1. Token and Vietnamese tone embeddings are combined.
2. Prelude transformer blocks project the input into a latent stream.
3. A recurrent transformer core is applied for multiple reasoning steps.
4. Coda transformer blocks decode the final latent state.
5. Output heads predict the next token, tone, and optional abstention score.

Core components:

- Grouped Query Attention with RoPE.
- Sandwich normalization and LayerScale for recurrent stability.
- Optional Mixture of Experts feed-forward blocks.
- Tonal tokenizer support for Vietnamese diacritics.
- Abstention head for later calibration against hallucination.

## Install

```bash
pip install -r requirements.txt
```

Python 3.9 or newer is required. CUDA is optional, but real training runs need a GPU.

## Bigram Tensor 1 on Windows Server

For the Windows Server + 1x RTX A6000/Blackwell 48GB deployment path, use the Tensor 1 guide:

```powershell
powershell -ExecutionPolicy Bypass -File run_windows_quickstart.ps1
```

Full docs: [docs/TENSOR1_WINDOWS_SERVER.md](docs/TENSOR1_WINDOWS_SERVER.md).

## Run Tests

```bash
python tests/run_all.py
```

For the model-only tests:

```bash
python tests/test_model.py
```

## Train A Tokenizer

```bash
python scripts/train_tokenizer.py \
  --input data/corpus.txt \
  --output data/tokenizer.json \
  --vocab-size 32000 \
  --min-frequency 2
```

## Prepare Binary Data

```bash
python scripts/prepare_data.py \
  --tokenizer data/tokenizer.json \
  --input data/train.txt \
  --output-prefix data/train

python scripts/prepare_data.py \
  --tokenizer data/tokenizer.json \
  --input data/val.txt \
  --output-prefix data/val
```

## Pretrain

Use a config file from `configs/` or a preset supported by the training script.

```bash
python scripts/train.py \
  --train-data data/train \
  --val-data data/val \
  --tokenizer data/tokenizer.json \
  --config configs/tiny.json \
  --max-steps 1000 \
  --out-dir checkpoints/pretrain
```

## Supervised Fine-Tuning

SFT data is JSONL with one object per line:

```json
{"prompt": "Xin chào!", "response": "Chào bạn, mình là Bigram."}
```

Run SFT:

```bash
python scripts/train_sft.py \
  --data data/sft.jsonl \
  --val-data data/sft_val.jsonl \
  --tokenizer data/tokenizer.json \
  --init checkpoints/pretrain/ckpt_final.pt \
  --out-dir checkpoints/sft
```

## Generate

```bash
python scripts/generate.py \
  --checkpoint checkpoints/sft/ckpt_final.pt \
  --tokenizer data/tokenizer.json \
  --prompt "Thủ đô Việt Nam là" \
  --recurrence 8 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --max-new-tokens 80
```

Generation knobs:

- `--recurrence`: number of recurrent reasoning steps.
- `--temperature`: sampling sharpness.
- `--top-k`: keep only the highest-k token logits.
- `--top-p`: nucleus sampling threshold.
- `--repetition-penalty`: discourages repeating prompt or generated tokens.
- `--abstention-threshold`: stop generation when the calibrated abstention head is not confident enough.

## Configuration

The main model fields live in `bigram/config.py` and JSON files under `configs/`.

Useful recurrent-depth settings:

- `mean_recurrence`: average recurrence used during training when no explicit recurrence is passed.
- `backprop_depth`: number of final recurrent steps kept in the gradient graph.
- `state_init_mode`: `zeros` for deterministic inference, `normal` for the legacy stochastic start.
- `recurrent_early_exit_tol`: disabled at `0.0`; set a positive tolerance to allow eval-time early exit.
- `recurrent_early_exit_min_steps`: minimum recurrent steps before early exit is considered.

## Nano Models

Nano checkpoints and tokenizers are stored outside the Python package layout:

- `nano1/`: Bigram Nano 1 local artifacts.
- `nano2/`: Bigram Nano 2 experimental router/synthesizer artifacts.
- `models/`: local model cards and publishing notes.

The public Nano 1 release is available at:

```text
https://huggingface.co/aevynt/bigram-nano-1
```

Example Nano 1 chat command:

```bash
python scripts/chat_nano1.py \
  --checkpoint nano1/sft/ckpt_final.pt \
  --tokenizer nano1/tokenizer.json
```

Example Nano 2 pipeline command:

```bash
python scripts/pipeline_nano2.py \
  --model-a nano2/model_a/sft/ckpt_final.pt \
  --model-b nano2/model_b/sft/ckpt_final.pt \
  --tokenizer nano2/tokenizer.json
```

## Repository Layout

```text
bigram/
  bigram/        Python package
  configs/       model and training configs
  data/          local datasets and JSONL files
  models/        model cards
  scripts/       training, data, generation, and pipeline scripts
  tests/         unit and mechanism tests
```

## Notes

This repository validates the training mechanics and architecture. Model quality still depends on data quality, training duration, and hardware. For production-like behavior, train on a larger clean corpus, hold out a real validation set, and run task-specific evaluations before publishing checkpoints.

## License

See [LICENSE](LICENSE).

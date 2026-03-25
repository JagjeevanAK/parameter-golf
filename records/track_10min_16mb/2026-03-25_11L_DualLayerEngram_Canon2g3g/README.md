# 11L Dual-Layer Engram + Canonical 2/3-gram Memory

Record candidate built on the `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` stack.

This folder is an implementation artifact, not a claimed leaderboard result yet. The script is wired and syntax-checked; full GPU training, export, and score confirmation are still pending.

## Goal

Replace the input-only `BigramHashEmbedding` path with a selected-layer Engram memory module while preserving the rest of the strong `1.1248` stack:

- 11-layer U-Net backbone
- XSA on the last 4 layers
- EMA
- Partial RoPE
- LN scale
- Mixed int6/int8 export path
- SmearGate

## What Changed

1. Added tokenizer canonicalization via SentencePiece piece normalization:
   - strip leading `▁`
   - `unicodedata.normalize("NFKC", piece).casefold()`
   - control / unknown / unused / byte pieces map to themselves
2. Added precomputed canonical 2-gram and 3-gram hashes inside `GPT.forward` and `GPT.forward_logits`.
3. Added `EngramModule` at layers `1` and `5` by default.
4. Replaced input-side `BigramHashEmbedding` with dual-order hashed memory retrieval inside selected blocks.
5. Kept Engram tables on the token/embed optimizer group, Engram projections on Muon, and mixer controls on scalar AdamW.
6. Forced Engram weights onto the existing int8 export path even when individual tensors are smaller than the default passthrough threshold.

## Engram Default Configuration

- Layers: `1,5`
- Orders: `2,3`
- Hash heads per order: `2`
- Head dim: `16`
- Buckets:
  - 2-gram: `1531,1543`
  - 3-gram: `3079,3089`
- Sentinel slot: last bucket entry per head
- Context gate: elementwise sigmoid between normalized hidden state and projected Engram key
- Mixer: causal depthwise shift-sum, `4` taps, dilation `3`
- Output scale init: `0.05`

## Novelty Claim

This is intentionally not another input-only n-gram table. The combination here is:

- selected-layer Engram memory
- tokenizer canonicalization before hashing
- contextual gating inside the transformer stack
- dense XSA backbone from the existing strong record

It is also intentionally not memory tokens or Titans-style neural memory.

## Default Run Command

```bash
NUM_LAYERS=11 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
ENGRAM_ENABLED=1 ENGRAM_LAYERS=1,5 ENGRAM_ORDERS=2,3 \
ENGRAM_HEADS_PER_ORDER=2 ENGRAM_HEAD_DIM=16 \
ENGRAM_ORDER2_BUCKETS=1531,1543 ENGRAM_ORDER3_BUCKETS=3079,3089 \
ENGRAM_MIXER_TAPS=4 ENGRAM_MIXER_DILATION=3 ENGRAM_OUT_SCALE=0.05 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Suggested Ablations

- `ENGRAM_ENABLED=0` for base-stack comparison
- `ENGRAM_LAYERS=1` for single-layer Engram
- `ENGRAM_MIXER_TAPS=0` to disable the depthwise mixer
- `ENGRAM_ORDER3_BUCKETS=` plus `ENGRAM_ORDERS=2` for order-2 only

## Current Validation Status

- `python3 -m py_compile train_gpt.py`: passed
- Full CUDA training / export / eval: pending user GPU run

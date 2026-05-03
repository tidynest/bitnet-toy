# Training Guide

How to train, what to expect, what to watch for, and what knobs to turn.

## Smoke test (no setup required)

```sh
cargo run --release
```

Runs M4 (1D linear regression), M6 (2D BitLinear regression), M9 (BitNet LM
on the 167-char `TINY_CORPUS`), and M10 (greedy generation + ternary export).
All four finish in a few seconds. Useful for verifying nothing is broken
before committing to a longer run.

## Real training: TinyShakespeare

### One-time setup

```sh
mkdir -p data
curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -o data/tinyshakespeare.txt
```

The file is ~1.1 MB, 65 unique characters, ~1.1 million sliding windows at
`seq_len = 64`.

### Run

```sh
cargo run --release -- shakespeare
```

Default config (defined in `TrainConfig::shakespeare()` in `src/main.rs`):

| Knob | Default | Notes |
|---|---|---|
| `n_steps`         | 5000      | ~2-3 min on CPU |
| `peak_lr`         | 3e-3      | After 200-step warmup |
| `floor_lr`        | 3e-4      | Cosine decay endpoint |
| `warmup_steps`    | 200       | Linear ramp from 0 to peak |
| `grad_clip`       | 1.0       | Global L2 norm cap |
| `weight_decay`    | 0.1       | AdamW decoupled |
| `adamw_beta1`     | 0.9       | First-moment decay |
| `adamw_beta2`     | 0.95      | Second-moment decay (LLaMA pick) |
| `seed`            | 1337      | LCG seed for init + window sampling |
| `model.hidden_dim`     | 64   | |
| `model.head_dim`       | 64   | Single-head; equals hidden_dim |
| `model.ffn_dim`        | 128  | 2x hidden_dim |
| `model.max_seq_len`    | 64   | Window length |
| `model.n_blocks`       | 4    | Transformer blocks |
| `log_every`            | 100  | Print status every N steps |

After the run, `models/shakespeare.ternary_packed.bin` contains the trained
model in base-3 packed format (~60 KB).

### Resume

Pass the checkpoint path as the second CLI argument:

```sh
cargo run --release -- shakespeare models/shakespeare.ternary_packed.bin
```

The trainer:
1. Imports the checkpoint via `export::import`.
2. Overrides the model config with the checkpoint's config (vocab, dims,
   block count must match).
3. Builds a *fresh* AdamW optimiser (its `m`, `v` are not persisted across
   runs).
4. Restarts the cosine LR schedule, including the warmup.

Expect the first ~30 steps after resume to be wobbly while AdamW's moment
estimates re-establish. After that, training continues normally from where
the checkpoint left off, but with slightly different per-step trajectories
than if training had run uninterrupted.

## Watching the run

Each log line looks like:

```
step  1500   train_loss = 2.7379   anchor_loss = 2.7096   min_seen = 2.0854   lr = 2.5401e-3   |g| = 1.549
```

| Column | Meaning |
|---|---|
| `train_loss`  | Loss on the random window this step. Noisy by nature. |
| `anchor_loss` | Loss on a fixed window (the first one). The smooth signal. |
| `min_seen`    | Best `train_loss` ever seen on any step. |
| `lr`          | Current learning rate (warmup or cosine decay). |
| `|g|`         | Pre-clip global L2 norm of all gradients. |

### Healthy convergence (on Shakespeare)

- `anchor_loss` starts ~4.18 (= log of 65 vocab).
- After warmup completes (step 200), drops steadily.
- Around step 1000-3000, plateaus in the 2.4-2.7 range while learning
  long-range structure.
- By step 5000, `min_seen` is in the 1.7-2.0 range with `anchor_loss`
  slightly above.
- `|g|` stays in 1.0-3.0 range. Brief spikes after warmup are fine.

### Warning signs

| Symptom | Likely cause | Counter-move |
|---|---|---|
| Loss collapses to <0.1 | Causal mask broken; model is "cheating" by reading the future | Confirm `attention.rs` calls `.causal_mask()` before softmax |
| Loss flat at log(vocab) | Gradient flow broken (zero grad on weights) | Run `cargo test`; the M9 gate would also fail |
| Loss diverges (NaN) | LR too high, or quantiser numerics | Lower `peak_lr` to 1e-3 |
| Loss bumpy but trending down | Normal STE noise | Tolerate; watch `anchor_loss`, not `train_loss` |
| `|g|` consistently above 5 | Underflow of grad clip, or LR too high | Lower `grad_clip` to 0.5 or `peak_lr` to 1e-3 |

## Generation modes

After training finishes, three generation modes print:

1. **Greedy** (deterministic argmax). Tends to fall into "the the the" loops
   because the most-likely-token chain has fixed points.

2. **Temperature 0.8** (slightly sharper than the model's raw distribution).
   Mostly picks the top option, occasionally diverges. Best balance of
   coherence and variety.

3. **Temperature 1.0** (raw model distribution). More variety, looser
   structure.

A fourth call would run from the loaded `model.shakespeare.ternary_packed.bin`
to verify the round-trip. The generation with the loaded weights is similar
but not identical to the in-memory model: ternary serialisation discards the
f32 master precision, so the post-load `Model` produces slightly different
logits. Output should still be sensible.

## Tuning

Edit `TrainConfig::shakespeare()` in `src/main.rs`. Common adjustments:

| Goal | Change |
|---|---|
| Train longer for better quality | `n_steps: 20_000` |
| Bigger model | `hidden_dim: 128, ffn_dim: 256` (slower) |
| Different style | Replace `data/tinyshakespeare.txt` with another corpus |
| Lower memory pressure | `max_seq_len: 32` |
| Different sampling temperature in demo | Edit the values inside `run_shakespeare_training` |

For new corpora, point `corpus_path` at a different UTF-8 file. Vocab is
rebuilt from whatever text is in that file. If the new vocab differs from
the previous run's, you cannot resume from that previous checkpoint.

## Limitations to know about

- The corpus must fit in memory. For TinyShakespeare (1.1 MB) this is fine;
  for a 1 GB corpus you'd want streamed window iteration.
- No batching: each step processes one window. Wall-clock training time
  scales linearly with `n_steps`.
- No validation split. If you want held-out perplexity, slice the corpus
  manually before encoding.
- AdamW state does not persist across `cargo run` invocations. This is OK
  for short resumes; for long resumed sessions, the warmup-and-no-momentum
  start can cost a few hundred steps of progress.

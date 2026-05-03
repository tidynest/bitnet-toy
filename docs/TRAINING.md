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
| `model.n_heads`        | 4    | Multi-head attention; sums of n_heads paths |
| `model.head_dim`       | 16   | Per-head dim; n_heads * head_dim == hidden_dim |
| `model.ffn_dim`        | 128  | 2x hidden_dim |
| `model.max_seq_len`    | 64   | Window length |
| `model.n_blocks`       | 4    | Transformer blocks |
| `log_every`            | 100  | Print status every N steps |
| `val_split`            | 0.10 | Tail fraction of corpus held out |
| `eval_every`           | 500  | Run val pass every N training steps |
| `val_eval_samples`     | 100  | Val windows per eval pass |
| `batch_size`           | 4    | Windows processed per optimiser step (averaged grads) |
| `n_workers`            | 4    | Threads used for parallel forward+backward across the batch |

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

Most log lines look like this:

```
step  1500   train_loss = 2.7379   anchor_loss = 2.7096   min_seen = 2.0854   lr = 2.5401e-3   |g| = 1.549
```

On `eval_every`-aligned steps (and the very last step), the line carries
extra columns from a held-out validation pass:

```
step   500   train_loss = ...   anchor_loss = ...   min_seen = ...   val_loss = 2.45   val_ppl = 11.6   lr = ...   |g| = ...
```

After the loop ends, a more accurate final pass with 5x the running-eval
sample count prints:

```
final validation:  val_loss = 2.31   val_ppl = 10.07   (uniform-vocab baseline = 65.0, ratio = 0.155)
```

| Column | Meaning |
|---|---|
| `train_loss`  | Loss on the random window this step. Noisy by nature. |
| `anchor_loss` | Loss on a fixed window (the first one). The smooth training signal. |
| `min_seen`    | Best `train_loss` ever seen on any step. |
| `val_loss`    | Mean cross-entropy across `val_eval_samples` held-out windows (deterministic stride). |
| `val_ppl`     | `exp(val_loss)`. Interpret as "model is as confused as if uniformly choosing among `val_ppl` options." Lower is better. |
| `lr`          | Current learning rate (warmup or cosine decay). |
| `|g|`         | Pre-clip global L2 norm of all gradients. |

`val_ppl` is the metric that actually matters for comparing architectural
variants. `train_loss` and `anchor_loss` are too noisy: STE quantisation
makes per-step training loss swing 0.5+ in either direction, and the anchor
window is a single point estimate. `val_ppl` averages over many held-out
windows the model never trained on, so it tracks generalisation rather
than memorisation.

Random-baseline `val_ppl` for char-vocab 65 is exactly 65 (uniform
distribution). A real char-LM gets `val_ppl` in the 3-12 range depending
on model size and training time.

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
- AdamW state IS persisted across `cargo run` invocations as of v0.7
  (BNT3 + OPTM payload). Old pre-v0.7 BNT3 checkpoints still load but
  the optimiser starts at zero momentum. The cosine LR schedule still
  restarts each run.
- Batching is window-level only. The forward pass for a single window is
  still serial across positions and matmul rows; if you want to actually
  use all 16 threads, push `batch_size` and `n_workers` up to 8 or 16.
  Future work: parallel matmul rows (TODO), then SIMD intrinsics, then
  GPU back-end via cudarc.

## Batching and threading notes

`batch_size = 4, n_workers = 4` is a deliberately moderate default for the
7940HS. Each thread handles one window's forward + backward; on a hot
laptop this peaks ~50 percent of the chip during the compute window and
drops back to idle during the optimiser update. If your fans tolerate it,
`batch_size = 8, n_workers = 8` doubles the per-step compute and roughly
halves the wall-clock time per step. `batch_size = 1` reverts to the
single-window deterministic path.

Setting `n_workers = 1` while `batch_size > 1` runs the batch serially in
one thread. This is byte-for-byte deterministic given the same RNG state -
useful for debugging or reproducible measurements.

Larger batches give smoother gradient estimates (less stochastic noise per
step) but each step costs `batch_size`x compute. The standard "linear LR
scaling" rule says LR should scale with batch_size for equivalent step
sizes, but our STE quantisation noise dominates at small batch sizes, so
the rule applies less cleanly here. Default LR is left unchanged across
batch sizes; tune empirically if you push to batch_size = 16+.

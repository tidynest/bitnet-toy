# bitnet-toy

A hand-rolled implementation of [BitNet b1.58](https://arxiv.org/abs/2402.17764)
in pure Rust. Every line written from scratch as a learning exercise: tensor
type, autograd, ternary quantisation, transformer block, training loop,
inference, binary export. No third-party ML dependencies.

## Status

- **93** tests passing on `cargo test`.
- **0** warnings on `cargo build --release`.
- `cargo audit` clean (stdlib-only crate; no transitive dependencies).
- Trains end-to-end on the full TinyShakespeare corpus in ~2-3 minutes on CPU.

## Quick start

```sh
# All M4-M10 demos (small models, ~2 seconds total).
cargo run --release

# Train on TinyShakespeare. Download the corpus first:
mkdir -p data
curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -o data/tinyshakespeare.txt
cargo run --release -- shakespeare

# Continue training from a previous checkpoint.
cargo run --release -- shakespeare models/shakespeare.ternary_packed.bin
```

After Shakespeare training, prints loss curve, then greedy + two temperature-sampling
generations from the prompts `ROMEO:`, `To be `, `King `. Saves the trained model to
`models/shakespeare.ternary_packed.bin`.

## What this is

Implements **BitNet b1.58**: ternary weights (`{-1, 0, +1}`) plus per-row INT8
activations, with a straight-through-estimator backward pass making the model
trainable despite the discrete forward.

The model is a small transformer (RMSNorm + scaled-dot-product attention with
causal mask + FFN, two blocks for the demo, four for Shakespeare) on a
character-level vocabulary. Generation is greedy or temperature-sampled. Trained
weights are exported in three on-disk formats; the smallest is **6x smaller**
than f32 baseline.

## CLI

```text
cargo run --release                          # M4-M10 demos
cargo run --release -- shakespeare           # fresh Shakespeare training
cargo run --release -- shakespeare <path>    # resume training from checkpoint
```

## Project layout

```
bitnet-toy/
├── data/        gitignored, holds training corpora
├── models/      gitignored, holds saved checkpoints
├── src/         every Rust source file
├── docs/
│   ├── ARCHITECTURE.md    file-by-file walkthrough
│   └── TRAINING.md        training guide and hyperparameter notes
├── README.md    this file
├── TODO.md      improvement queue and test-training plan
├── Cargo.toml
└── Cargo.lock
```

## Source map

| File | Role |
|---|---|
| `src/tensor.rs`      | row-major f32 tensor, matmul, transpose, elementwise ops |
| `src/autograd.rs`    | tape-based reverse-mode autograd, STE quantiser ops, causal mask |
| `src/bitlinear.rs`   | `absmean_ternary` and `absmax_int8` quantiser primitives |
| `src/attention.rs`   | single-head scaled-dot-product self-attention |
| `src/ffn.rs`         | position-wise feed-forward network |
| `src/block.rs`       | transformer block (norm + attention + FFN + residuals) |
| `src/model.rs`       | `Model`, `ModelConfig`, parameter visitor, init via LCG |
| `src/data.rs`        | char vocab, sliding windows, file loader, LCG, shuffler |
| `src/optim.rs`       | AdamW, gradient clipping, cosine LR with warmup |
| `src/inference.rs`   | greedy + temperature autoregressive generation |
| `src/export.rs`      | three binary formats with header + round-trip importer |
| `src/main.rs`        | CLI dispatch, `TrainConfig`, demos, integration tests |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what each module exposes
and how the pieces compose.

## Training

Brief recipe (full guide in [docs/TRAINING.md](docs/TRAINING.md)):

1. Place a UTF-8 corpus at `data/tinyshakespeare.txt` (or any path you set in
   `TrainConfig.corpus_path`).
2. Run `cargo run --release -- shakespeare`.
3. Watch the four-column status line every 100 steps:
   `train_loss`, `anchor_loss` (smooth signal), `min_seen`, `lr`, `|g|`.
4. After 5000 steps the model is exported to `models/shakespeare.ternary_packed.bin`.

To resume, pass that file path back as the second CLI argument; training
continues from the loaded weights with a fresh AdamW optimiser state.

## Export formats

| Format | Per-weight cost | Compression vs f32 (M9 demo model) |
|---|---|---|
| Float32     | 4 bytes      | 1.00x |
| Ternary i8  | 1 byte       | 2.92x |
| TernaryPacked (base-3, 5 per byte) | ~1.6 bits | 6.02x |

Embeddings (`token_embed`, `pos_embed`) stay f32 in all formats. Block weights
and the LM head are quantised in the two ternary formats. The packed format
uses base-3 encoding (`3^5 = 243 < 256`) to fit five ternary values per byte.

The importer in `export::import` reads any of the three formats, returning a
`Model` plus the `Format` it was stored as.

## Build, test, audit

```sh
cargo build --release       # optimised binary at target/release/bitnet-toy
cargo test                  # runs 93 tests
cargo fmt                   # apply rustfmt
cargo clippy --all-targets  # extra lints (pedantic warnings allowed at crate level)
cargo audit                 # security audit; trivially clean (no deps)
```

## Constraints

This is a learning project, not a production library:

- Pure Rust, no third-party ML dependencies (only `std`).
- Single-threaded, no SIMD intrinsics, no GPU.
- f32 throughout; no BF16 or FP16.
- Single-head attention (multi-head deferred, see [TODO.md](TODO.md)).
- No KV cache; inference recomputes the full forward per token.

For production needs, use [Burn](https://burn.dev), [candle](https://github.com/huggingface/candle),
or [tch-rs](https://github.com/LaurentMazare/tch-rs).

## Further reading

- **[TODO.md](TODO.md)** queue of pending improvements and the test-training plan.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** how the modules compose.
- **[docs/TRAINING.md](docs/TRAINING.md)** training recipe with hyperparameter rationale.
- **[BitNet b1.58 paper](https://arxiv.org/abs/2402.17764)** the architecture this implements.
- **[Karpathy's char-rnn post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)** for context on
  what character-level language modelling looks like at this scale.

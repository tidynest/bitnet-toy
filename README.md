# bitnet-toy

A hand-rolled implementation of [BitNet b1.58](https://arxiv.org/abs/2402.17764)
in pure Rust. Every line written from scratch as a learning exercise: tensor
type, autograd, ternary quantisation, transformer block, training loop,
inference, binary export. No third-party ML dependencies.

## Status

- **119** tests passing on `cargo test`.
- **0** warnings on `cargo build --release`.
- `cargo audit` clean (stdlib-only crate; no transitive dependencies).
- Trains end-to-end on the full TinyShakespeare corpus in ~8-15 minutes on CPU (v0.9 ~2M-param config).

## Quick start

```sh
# All M4-M10 demos (small models, ~2 seconds total).
cargo run --release

# Train on TinyShakespeare. Download the corpus first:
mkdir -p data
curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -o data/tinyshakespeare.txt
cargo run --release -- shakespeare

# Continue training from a previous checkpoint. The .f32.bin file preserves
# every master weight (and the AdamW optimiser state) byte-identical, so a
# resume picks up exactly where the run ended:
cargo run --release -- shakespeare models/shakespeare.f32.bin
```

After Shakespeare training, prints loss curve, then greedy + three sampled-mode
generations (Temperature, top-k, top-p) from the prompts `ROMEO:`, `To be `,
`King `. Saves two artefacts:

- `models/shakespeare.f32.bin`               full-precision masters + optim state. Use for clean resume.
- `models/shakespeare.ternary_packed.bin`    base-3 packed deployment artefact (~50x smaller).

## What this is

Implements **BitNet b1.58**: ternary weights (`{-1, 0, +1}`) plus per-row INT8
activations, with a straight-through-estimator backward pass making the model
trainable despite the discrete forward.

The model is a small transformer (RMSNorm + multi-head scaled-dot-product
attention with RoPE + causal mask + SwiGLU FFN, two blocks for the demo,
six for Shakespeare) on a character-level vocabulary. Generation supports
greedy, temperature, top-k, and top-p (nucleus) sampling. Trained weights
are exported in three on-disk formats; the smallest is **6x smaller** than
f32 baseline.

## CLI

```text
cargo run --release                                  # M4-M10 demos
cargo run --release -- shakespeare                   # fresh Shakespeare training (~5M params)
cargo run --release -- shakespeare <path>            # resume ~5M training from checkpoint
cargo run --release -- shakespeare-large             # fresh ~8.5M training (seq_len 128)
cargo run --release -- shakespeare-large <path>      # resume ~8.5M training from checkpoint
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
4. After 5000 steps the model is exported to both `models/shakespeare.f32.bin`
   (lossless, for resume) and `models/shakespeare.ternary_packed.bin` (compact).

To resume, pass `models/shakespeare.f32.bin` as the second CLI argument;
training continues from the same masters and the same AdamW momentum,
so the first step's val_ppl matches where the previous run ended.
Resuming from the packed file still works but pays roughly 500 wasted
steps re-establishing the master values from `γ · W_q`.

## Export formats

| Format | Per-weight cost | Compression vs f32 (M9 demo model) |
|---|---|---|
| Float32 (with masters)             | 4 bytes      | 1.00x   (use for resume) |
| Ternary i8                         | 1 byte       | 2.92x |
| TernaryPacked (base-3, 5 per byte) | ~1.6 bits    | 6.02x   (use for distribution) |

Embeddings (`token_embed`, `pos_embed`) stay f32 in all formats. Block weights
and the LM head are quantised in the two ternary formats. The packed format
uses base-3 encoding (`3^5 = 243 < 256`) to fit five ternary values per byte.

The importer in `export::import` reads any of the three formats, returning a
`Model` plus the `Format` it was stored as.

## Build, test, audit

```sh
cargo build --release       # optimised binary at target/release/bitnet-toy
cargo test                  # runs 119 tests
cargo fmt                   # apply rustfmt
cargo clippy --all-targets  # extra lints (pedantic warnings allowed at crate level)
cargo audit                 # security audit; trivially clean (no deps)
```

## Constraints

This is a learning project, not a production library:

- Pure Rust, no third-party ML dependencies (only `std`).
- f32 throughout; no BF16 or FP16.
- SIMD inside `Tensor::matmul` on x86_64, runtime-detected, widest path
  first: AVX-512 foundation (16 f32 per inner-loop step) on Zen 4 /
  Sapphire Rapids and later, falling back to AVX2 (8 f32) and then to a
  scalar AXPY. All three are bit-identical per output cell because none
  use FMA. Multi-threaded across output rows via `std::thread::scope`.
  Empirical note: AVX-512 underperforms AVX2 on Zen 4
  (memory-bandwidth bound); export `BITNET_MATMUL_SIMD=avx2` to opt
  out of AVX-512 there.
- Optional CUDA back-end (`--features cuda`): hand-rolled NVRTC
  tile-based GEMM kernel via `cudarc 0.19`. Phase 1 surface in
  `src/cuda.rs` is matmul-only (`CudaTensor` + `cuda_matmul`); the
  training pipeline still runs on CPU. NOT bit-identical to CPU
  (parallel reduction reorders sums) but agrees within `1e-4 + 1e-4 *
  |val|`. `cargo run --release --features cuda -- cuda-demo` shows the
  per-call CPU-vs-GPU microbench.
- KV cache for inference (`src/inference_kv.rs`) gives roughly 50-100x
  faster per-token generation; the older `inference::generate_with_mode`
  recomputes the full forward each step and is kept for parity testing.

For production needs, use [Burn](https://burn.dev), [candle](https://github.com/huggingface/candle),
or [tch-rs](https://github.com/LaurentMazare/tch-rs).

## Further reading

- **[TODO.md](TODO.md)** queue of pending improvements and the test-training plan.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** how the modules compose.
- **[docs/TRAINING.md](docs/TRAINING.md)** training recipe with hyperparameter rationale.
- **[BitNet b1.58 paper](https://arxiv.org/abs/2402.17764)** the architecture this implements.
- **[Karpathy's char-rnn post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)** for context on
  what character-level language modelling looks like at this scale.

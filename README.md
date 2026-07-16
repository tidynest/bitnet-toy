# bitnet-toy

![version](https://img.shields.io/badge/version-v0.19.0-3b6ea5)
![Rust](https://img.shields.io/badge/Rust-2024%20edition-ce412b?logo=rust&logoColor=white)
![tests](https://img.shields.io/badge/tests-140%20passing-3f9142)
![CUDA](https://img.shields.io/badge/CUDA-optional%20(cudarc%200.19)-76b900?logo=nvidia&logoColor=white)
![ML dependencies](https://img.shields.io/badge/ML%20dependencies-none-3f9142)

A hand-rolled implementation of [BitNet b1.58](https://arxiv.org/abs/2402.17764)
in pure Rust. Every line written from scratch as a learning exercise: tensor
type, autograd, ternary quantisation, transformer block, training loop,
inference, binary export. No third-party ML dependencies.

## Contents

- [Status](#status)
- [Quick start](#quick-start)
- [What this is](#what-this-is)
- [CLI](#cli)
- [Project layout](#project-layout)
- [Source map](#source-map)
- [Training](#training)
- [Export formats](#export-formats)
- [Build, test, audit](#build-test-audit)
- [Constraints](#constraints)
- [Roadmap](#roadmap)
- [Further reading](#further-reading)

## Status

- **140** tests passing on `cargo test`; **172** with `cargo test --features cuda`.
- **0** warnings on `cargo build --release` (or `--features cuda`).
- `cargo audit` clean for the default build (stdlib-only); the optional
  `cuda` feature pulls `cudarc` and its small dynamic-loading deps.
- Trains end-to-end on the full TinyShakespeare corpus in ~8-15
  minutes on CPU (v0.13 ~5M-param config); current best **val_ppl
  4.869** at 30k cumulative steps.
- **Real BitNet ternary training runs on the GPU through Ada
  tensor cores.** Phase 5.a added STE quant kernels and a
  `BitLinear` trait; Phase 5.b rewrote the `CudaTensor` impl to
  use `cublasGemmEx` int8 GEMM (CUDA_R_8I / CUDA_R_32I /
  CUBLAS_COMPUTE_32I / CUBLAS_GEMM_DEFAULT_TENSOR_OP) with a
  shape-fallback to f32 sgemm when stride alignment fails (only
  hit by lm_head's `n = vocab = 65`). All Phase 5.a tests pass
  with the int8 path active; cuda-shakespeare runs end-to-end at
  v0.13 scale (loss trajectory matches CPU). Empirically GPU is
  ~300 ms/step vs CPU ~180 ms/step at v0.13 scale: per-step
  launch overhead dominates (~3000+ kernel launches per step).
  Kernel fusion + CUDA graphs + larger batches are the
  follow-ups to actually realise tensor-core throughput.

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

# Optional GPU back-end (Phase 1-3). Requires the CUDA toolkit on the host
# (Arch: `sudo pacman -S cuda`, lives at /opt/cuda). Build + run the
# end-to-end forward microbench:
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo run --release --features cuda -- cuda-forward-bench
```

After Shakespeare training, prints loss curve, then greedy + three sampled-mode
generations (Temperature, top-k, top-p) from the prompts `ROMEO:`, `To be `,
`King `. Saves two artefacts:

- `models/shakespeare.f32.bin`               full-precision masters + optim state. Use for clean resume.
- `models/shakespeare.ternary_packed.bin`    base-3 packed deployment artefact (~6x smaller).

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
cargo run --release -- --help                             # all subcommands + train flags
cargo run --release                                       # M4-M10 demos
cargo run --release -- train <corpus> [options]           # train any UTF-8 corpus; see `--help` for
                                                          #   --steps --lr --batch-size --seed --hidden --ffn
                                                          #   --heads --head-dim --blocks --seq-len --val-split
                                                          #   --out --resume --cuda
cargo run --release -- shakespeare                        # fresh Shakespeare training (~5M params)
cargo run --release -- shakespeare <path>                 # resume ~5M training from checkpoint
cargo run --release -- shakespeare-large                  # fresh ~8.5M training (seq_len 128)
cargo run --release -- shakespeare-large <path>           # resume ~8.5M training from checkpoint
cargo run --release -- sample <path>                      # skip training; print samples on the 3 default prompts
cargo run --release -- sample <path> <prompt...>          # skip training; sample from a caller-supplied prompt
cargo run --release -- sample <path> --corpus <corpus> .. # rebuild vocab from a custom training corpus
BITNET_SAMPLE_MODES=min cargo run --release -- sample ... # only the 2 highest-signal modes (top-p T=0.5 + KV-cache)
BITNET_SAMPLE_MODES=topp_low,kv cargo run -- shakespeare  # subset; same env var also gates the post-train tail
cargo run --release --features cuda -- cuda-demo          # CPU-vs-cuBLAS matmul microbench
cargo run --release --features cuda -- cuda-forward-bench # CPU-vs-CudaModel end-to-end forward bench
cargo run --release --features cuda -- cuda-train-demo    # Phase 4 end-to-end GPU training proof-of-concept
cargo run --release --features cuda -- cuda-shakespeare    # Phase 5.a real BitNet ternary training on GPU
cargo run --release --features cuda -- cuda-shakespeare-large  # Phase 5.a, ~8.5M-param config, seq_len 128
```

Training an arbitrary corpus end to end:

```text
cargo run --release -- train data/my_corpus.txt --steps 2000 --out mymodel
cargo run --release -- train data/my_corpus.txt --resume models/mymodel.f32.bin --out mymodel
cargo run --release -- sample models/mymodel.f32.bin --corpus data/my_corpus.txt "once upon"
```

The char vocab is built from the corpus itself, so `sample` needs the same
corpus (via `--corpus`) to rebuild it; the vocab-size check catches a
mismatched file.

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
├── Cargo.toml
└── Cargo.lock
```

## Source map

| File | Role |
|---|---|
| `src/tensor.rs`        | row-major f32 tensor; matmul (AVX-512 / AVX2 / scalar; parallel); inherent helpers for every Phase 2 op |
| `src/autograd.rs`      | tape-based reverse-mode autograd, STE quantiser ops, RoPE, causal mask, RMSNorm, SiLU |
| `src/bitlinear.rs`     | `absmean_ternary` and `absmax_int8` quantiser primitives |
| `src/attention.rs`     | multi-head scaled-dot-product self-attention (sum-of-projections, causal mask, RoPE) |
| `src/ffn.rs`           | SwiGLU position-wise feed-forward network |
| `src/block.rs`         | transformer block (RMSNorm + attention + FFN + residuals) |
| `src/model.rs`         | `Model`, `ModelConfig`, parameter visitor, init via LCG |
| `src/data.rs`          | char vocab, sliding windows, file loader, LCG, shuffler |
| `src/optim.rs`         | AdamW, gradient clipping, cosine LR with warmup, resume continuation |
| `src/inference.rs`     | greedy / temperature / top-k / top-p autoregressive generation |
| `src/inference_kv.rs`  | KV-cached generator (~50-100x faster per-token vs full-forward path) |
| `src/export.rs`        | three binary formats with header + round-trip importer + AdamW state payload |
| `src/device.rs`        | per-op traits (`MatMul`, `Add`, `Mul`, `MulScalar`, `Transpose2D`, `Softmax`, `CausalMask`, `Rope`, `Silu`, `RmsNorm`); generic helpers `attention_head_inference<T>`, `ffn_inference<T>`, `block_inference<T>` |
| `src/cuda.rs`          | CUDA back-end (gated `--features cuda`): NVRTC kernels, cuBLAS sgemm + int8 `cublasGemmEx` tensor-core GEMM, `CudaTensor`, `CudaModel` forward + GPU ternary training |
| `src/main.rs`          | CLI dispatch, `TrainConfig`, demos, integration tests, `cuda-demo` + `cuda-forward-bench` benches |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what each module exposes
and how the pieces compose.

## Training

Brief recipe (full guide in [docs/TRAINING.md](docs/TRAINING.md)):

1. Place a UTF-8 corpus at `data/tinyshakespeare.txt` (or anywhere, and use
   `train <path>` instead).
2. Run `cargo run --release -- shakespeare` (preset) or
   `cargo run --release -- train <corpus> [options]` (any corpus,
   hyperparameters overridable from the command line).
3. Watch the four-column status line every 100 steps:
   `train_loss`, `anchor_loss` (smooth signal), `min_seen`, `lr`, `|g|`.
4. After training completes, the model is exported to both `models/shakespeare.f32.bin`
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

The embedding table (`token_embed`) stays f32 in all formats; with tied
embeddings it doubles as the (transposed) LM-head weight. Block weights are
quantised in the two ternary formats. The packed format uses base-3 encoding
(`3^5 = 243 < 256`) to fit five ternary values per byte.

The importer in `export::import` reads any of the three formats, returning a
`Model` plus the `Format` it was stored as.

## Build, test, audit

```sh
cargo build --release       # optimised binary at target/release/bitnet-toy
cargo test                  # runs 139 tests
cargo fmt                   # apply rustfmt
cargo clippy --all-targets  # extra lints (pedantic warnings allowed at crate level)
cargo audit                 # security audit; trivially clean (no deps)
```

## Constraints

This is a learning project, not a production library:

- Pure Rust, no third-party ML dependencies (only `std`).
- f32 throughout; no BF16 or FP16.
- SIMD inside `Tensor::matmul` on x86_64, runtime-detected, widest path
  that actually wins: AVX-512 foundation (16 f32 per inner-loop step) on
  Sapphire Rapids and later, falling back to AVX2 (8 f32) and then to a
  scalar AXPY. All three are bit-identical per output cell because none
  use FMA. Multi-threaded across output rows via `std::thread::scope`.
  On Zen 4 (CPUID family `0x19`) AVX-512 is double-pumped and
  memory-bandwidth bound, so it runs ~9% slower than AVX2; the dispatcher
  detects Zen 4 and auto-selects AVX2 there. Overrides:
  `BITNET_MATMUL_SIMD=avx512` forces AVX-512 back on (for A/B timing),
  `=avx2` forces AVX2 anywhere, `=off` forces scalar.
- Optional CUDA back-end (`--features cuda`) via `cudarc 0.19`. Phase
  1: cuBLAS sgemm matmul + 9 hand-rolled NVRTC kernels for the rest
  of the op set (add, mul, mul_scalar, transpose_2d, causal_mask,
  softmax, rope, silu, rmsnorm). Phase 2: per-op trait architecture
  (`src/device.rs`) so a single generic helper compiles + runs on
  both `Tensor` and `CudaTensor`. Phase 3: `CudaModel` end-to-end
  forward (embed -> blocks -> rmsnorm -> lm_head matmul, all
  device-resident). NOT bit-identical to CPU (parallel reduction
  reorders sums; cuBLAS picks its own internal tile schedule) but
  agrees within `1e-3` per-op, `5e-3` per-block, `2e-2` end-to-end.
  Training stays on CPU until Phase 4 lands per-op backward kernels.
  `cargo run --release --features cuda -- cuda-forward-bench` shows
  the end-to-end CPU-vs-GPU benchmark. GPU training landed in Phase 4
  (f32) and then Phase 5.a / 5.b (real ternary BitNet via int8 tensor
  cores, `cuda-shakespeare`); see the Status section.
- KV cache for inference (`src/inference_kv.rs`) gives roughly 50-100x
  faster per-token generation. Sliding-window since v0.16.1 (cache
  capped at `max_seq_len` rows; RoPE reapplied at logical position so
  output stays coherent for arbitrarily long generations). The older
  `inference::generate_with_mode` recomputes the full forward each
  step and is kept for parity testing.

For production needs, use [Burn](https://burn.dev), [candle](https://github.com/huggingface/candle),
or [tch-rs](https://github.com/LaurentMazare/tch-rs).

## Roadmap

Planned work is tracked on GitHub:

- **[Issues](https://github.com/tidynest/bitnet-toy/issues)** the active backlog.
- **[Roadmap board](https://github.com/users/tidynest/projects/7)** issues grouped by milestone.

Milestones, in priority order:

1. **Phase 5.c - GPU perf** - realise the tensor-core speedup. Device buffers
   are now reused across steps (`sync_from_cpu`, #1) and the quant kernels are
   fused to one launch per operand (#2); remaining: capture the step as a CUDA
   graph (#3), then benchmark at scale (#4).
2. **CPU SIMD & threading** - the persistent matmul thread pool landed (#7)
   and Zen 4 AVX2 auto-select in v0.19; remaining: an ARM64 NEON path (#6,
   needs ARM hardware to validate).
3. **CLI ergonomics** - done (#8): `train <corpus>` with overridable
   hyperparameters, `--help`, and `sample --corpus` for custom checkpoints.

## Further reading

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** how the modules compose.
- **[docs/TRAINING.md](docs/TRAINING.md)** training recipe with hyperparameter rationale.
- **[BitNet b1.58 paper](https://arxiv.org/abs/2402.17764)** the architecture this implements.
- **[Karpathy's char-rnn post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)** for context on
  what character-level language modelling looks like at this scale.

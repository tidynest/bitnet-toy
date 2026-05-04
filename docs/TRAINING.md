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
| `n_steps`         | 10000     | ~8-15 min on CPU (v0.9 ~2M-param config) |
| `peak_lr`         | 3e-3      | After 200-step warmup |
| `floor_lr`        | 3e-4      | Cosine decay endpoint |
| `warmup_steps`    | 200       | Linear ramp from 0 to peak; fixed budget regardless of n_steps |
| `grad_clip`       | 1.0       | Global L2 norm cap |
| `weight_decay`    | 0.1       | AdamW decoupled |
| `adamw_beta1`     | 0.9       | First-moment decay |
| `adamw_beta2`     | 0.95      | Second-moment decay (LLaMA pick) |
| `seed`            | 1337      | LCG seed for init + window sampling |
| `model.hidden_dim`     | 128  | Doubled at v0.9 to break through the 280k-param val_ppl ceiling |
| `model.n_heads`        | 8    | Multi-head attention; head_dim invariant kept by doubling head count |
| `model.head_dim`       | 16   | Per-head dim; n_heads * head_dim == hidden_dim |
| `model.ffn_dim`        | 256  | 2x hidden_dim |
| `model.max_seq_len`    | 64   | Window length |
| `model.n_blocks`       | 6    | Transformer blocks |
| `log_every`            | 100  | Print status every N steps |
| `val_split`            | 0.10 | Tail fraction of corpus held out |
| `eval_every`           | 500  | Run val pass every N training steps (= 20 evals over 10k steps) |
| `val_eval_samples`     | 100  | Val windows per eval pass |
| `batch_size`           | 4    | Windows processed per optimiser step (averaged grads) |
| `n_workers`            | 4    | Threads used for parallel forward+backward across the batch. v0.10 briefly switched this to 1 in favour of per-matmul threading; v0.11 restored it to 4 after measuring that per-matmul threading is a net loss at this model scale (matmuls too small for spawn overhead to amortise). |

After the run two artefacts land in `models/`:

| File | Format | Purpose |
|---|---|---|
| `shakespeare.f32.bin`            | `Float32` (with masters)  | Lossless resume. Every BitLinear master and the AdamW `m`/`v` buffers are saved verbatim. |
| `shakespeare.ternary_packed.bin` | `TernaryPacked` (base-3, 5/byte) | Compact deployment artefact. Re-quantises masters as `γ · W_q`. **Optim state is intentionally omitted** so the file stays close to the "1.58 bits per weight" theoretical minimum; resume always uses the `.f32.bin`. |

### Resume

Pass the lossless `.f32.bin` checkpoint as the second CLI argument:

```sh
cargo run --release -- shakespeare models/shakespeare.f32.bin
```

The trainer:
1. Imports the checkpoint via `export::import`.
2. Overrides the model config with the checkpoint's config (vocab, dims,
   block count must match).
3. Restores the AdamW `m`/`v` buffers from the checkpoint's `OPTM` payload.
4. Restarts the cosine LR schedule, including the warmup.

Resuming from `shakespeare.f32.bin` is a true continuation: the first
training step's loss matches the val_loss the previous run ended on,
because the masters and the optimiser moments are byte-identical to the
final state of the previous run. (The cosine LR schedule still restarts
from the warmup floor; that is intentional and gives every fresh `cargo
run` a clean, predictable trajectory.)

Resuming from `shakespeare.ternary_packed.bin` still works but pays
roughly 500 wasted steps re-establishing the master values from the
lossy `γ · W_q` decomposition. Visible as a "step 0 val_ppl" spike well
above the value the previous run ended on. Use the packed file for
distribution, the `.f32.bin` for development resumes.

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

### Healthy convergence (on Shakespeare, v0.9 ~2M-param config)

- `anchor_loss` starts ~4.18 (= log of 65 vocab).
- After warmup completes (step 200), drops steadily.
- Around step 1000-3000, plateaus in the 2.0-2.4 range while learning
  long-range structure.
- By step 10000, `min_seen` is expected in the 1.4-1.7 range with
  `anchor_loss` slightly above. Target `val_ppl` 4-6 range (vs the
  v0.5-v0.8 ~6.8 ceiling on the smaller 280k-param config).
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

A fourth call would run from a loaded checkpoint to verify the round-trip.
The output depends on which file is loaded: `models/shakespeare.f32.bin`
produces logits identical to the in-memory model (lossless masters), while
`models/shakespeare.ternary_packed.bin` produces logits that are similar
but not identical because ternary serialisation projects each master onto
`γ · W_q`. Output should still be sensible in either case.

## Tuning

Edit `TrainConfig::shakespeare()` in `src/main.rs`. Common adjustments:

| Goal | Change |
|---|---|
| Train longer for better quality | `n_steps: 20_000` |
| Smaller / faster model (v0.5-v0.8 config) | `hidden_dim: 64, ffn_dim: 128, n_heads: 4, n_blocks: 4, n_steps: 5_000` |
| Even bigger model | `hidden_dim: 256, ffn_dim: 512, n_heads: 16` (much slower) |
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

The project has two levels of parallelism available; **only one is on by
default**, picked by where each unit of work is large enough for
thread-spawn cost (~10-30 us per spawn) to amortise:

- **Per-window threading** (v0.5+, the default level): `compute_batched_grads`
  spawns up to `n_workers` threads via `std::thread::scope`, each running
  one window's full forward + backward pass. With `batch_size = 4` and
  `n_workers = 4` that is 4 thread spawns per step, ~40k spawns over a
  10k-step run. Per-thread work is large (~30 matmuls + the tape +
  cross-entropy + the gradient walk), so spawn cost is negligible.

- **Per-matmul threading** (v0.10+, opt-in): `Tensor::matmul` *can* shard
  its output rows across threads via `chunks_mut`. Output is bit-identical
  to the serial path because each thread owns disjoint output rows.
  Default is **1** (serial). Set `BITNET_MATMUL_THREADS=N>1` to opt in.
  Only worthwhile when individual matmuls are large (~100k+ output
  elements). At the v0.9 model scale (every matmul has m*n in the
  1k-16k range) per-matmul threading was measured ~10x slower than
  serial because the spawn overhead dominated; v0.10's default of
  `min(available_parallelism, 8)` was reverted to 1 in v0.11.

**Do not stack both levels** at the same time: with `n_workers = 4` and
`BITNET_MATMUL_THREADS = 4` you would have 4 outer threads each spawning
4 inner threads = 16 fresh OS threads per matmul on a 16-thread chip,
which is 100% over-subscription and starts paging out OS scheduling
time. Pick one level.

`batch_size = 4` stays as the default: it is the gradient-smoothing knob,
not the parallelism knob. Larger batches give smoother gradient estimates
(less stochastic noise per step) but each step costs `batch_size`x
compute. `batch_size = 1` reverts to the single-window deterministic
path.

The standard "linear LR scaling" rule says LR should scale with batch_size
for equivalent step sizes, but STE quantisation noise dominates at small
batch sizes, so the rule applies less cleanly here. Default LR is left
unchanged across batch sizes; tune empirically if you push to
batch_size = 16+.

### Tuning the matmul thread budget

```sh
BITNET_MATMUL_THREADS=4 cargo run --release -- shakespeare    # explicit 4 threads
BITNET_MATMUL_THREADS=1 cargo run --release -- shakespeare    # serial (debug or profiling)
BITNET_MATMUL_THREADS=8 cargo run --release -- shakespeare    # full subscription on a 16-thread chip
```

The variable is read once at first matmul call and cached for the rest
of the process. To change it for a fresh run, set the env var before
launching the binary.

### Tuning the matmul SIMD layer

Since v0.11 `Tensor::matmul`'s inner kernel is register-blocked AXPY
(loop order i, kk, j) so the innermost loop is contiguous
`out_row[j] += a * rhs_row[j]` over f32 slices. v0.17 layered AVX-512
foundation (`avx512f`) on top of the v0.11 AVX2 path; the dispatcher
picks the widest path the CPU exposes:

| Path     | Lanes | Detection                     |
|----------|-------|-------------------------------|
| AVX-512  |    16 | `is_x86_feature_detected!("avx512f")` (Zen 4, Sapphire Rapids onwards) |
| AVX2     |     8 | `is_x86_feature_detected!("avx2")` (everything since ~2013)            |
| Scalar   |     1 | always                                                                 |

All three paths produce **byte-identical** output per cell because
per-cell accumulation order (`kk = 0..k`) is unchanged and none of them
uses fused multiply-add. The dispatcher itself reads
`BITNET_MATMUL_SIMD` once, caches the result in a `OnceLock`, and stays
branch-free for the rest of the process.

```sh
cargo run --release -- shakespeare                              # widest available (AVX-512 on Zen 4)
BITNET_MATMUL_SIMD=avx2 cargo run --release -- shakespeare      # force AVX2 even on AVX-512 hardware
BITNET_MATMUL_SIMD=off  cargo run --release -- shakespeare      # force scalar (also: 0 | none | scalar)
```

Empirical 180-second smoke timings on a 7940HS (Zen 4, n_workers = 4)
on the v0.13 ~5M-param `shakespeare` config (the v0.11 numbers were on
the v0.9 ~2M-param config, so steps/sec is roughly 2x lower here):

| Config                     | Steps reached in 180s | Steps/sec |
|----------------------------|----------------------:|----------:|
| `BITNET_MATMUL_SIMD=avx2`  |                  1200 |      6.67 |
| AVX-512 (v0.17 default)    |                  1100 |      6.11 |
| `BITNET_MATMUL_SIMD=off`   |                  1100 |      6.11 |

Cross-mode bit-equality: every printed metric (`train_loss`,
`anchor_loss`, `min_seen`, `lr`, `|g|`) is byte-identical at every
shared step number across all three modes. The full training
pipeline - 1100 forward+backward+AdamW passes including RoPE,
SwiGLU, INT8 activation quant, ternary STE, batched gradient
averaging, grad clipping - produces the same f32 cells regardless
of which SIMD width was used.

**Surprise on Zen 4: AVX-512 ties scalar; AVX2 wins by ~9 percent.**
The plausible interpretations:

1. **Memory-bandwidth bound at this model scale.** The 16-wide
   AVX-512 loads pull more bytes per cycle per thread, increasing
   L2/L3 contention across the 4 worker threads. AVX2's 8-wide
   loads are gentler on the bus, so total throughput is higher
   when 4 threads each issue them concurrently.

2. **Zen 4 implements 512-bit ops as two 256-bit micro-ops.** Each
   `_mm512_mul_ps` is dispatched as 2 uops internally, doubling
   front-end pressure for no per-cycle throughput gain. The gain
   on Zen 4 was supposed to be the halved inner-loop trip count
   (so half the loop bookkeeping); evidently that gain is smaller
   than the dispatch cost.

3. **LLVM autovectorises the scalar AXPY** to SSE / AVX1 on
   `--release` builds, which is why "scalar" is not actually slow
   here. The "scalar" row is really "compiler-decided SIMD".

4. Some part is sample noise: 100-step granularity over 180s is a
   ~9 percent resolution. AVX-512 could be at step 1199.

On Intel Sapphire Rapids and later the 512-bit units are native
(not double-pumped) and the speedup over AVX2 should be larger
there - re-run on that hardware before making conclusions about
the AVX-512 path's general value. For now on Zen 4 the
`BITNET_MATMUL_SIMD=avx2` override is worth using for production
training runs.

To benchmark locally:

```sh
time timeout 180 ./target/release/bitnet-toy shakespeare > simd_avx512.log 2>&1
time BITNET_MATMUL_SIMD=avx2 timeout 180 ./target/release/bitnet-toy shakespeare > simd_avx2.log 2>&1
time BITNET_MATMUL_SIMD=off  timeout 180 ./target/release/bitnet-toy shakespeare > simd_off.log 2>&1
grep "^step" simd_avx512.log | tail -1
grep "^step" simd_avx2.log   | tail -1
grep "^step" simd_off.log    | tail -1

# Cross-mode bit-equality check - shared step numbers must match
# byte-for-byte. The fastest mode will have extra trailing lines past
# the slowest one's last log; those are expected.
diff <(grep "^step" simd_avx512.log) <(grep "^step" simd_avx2.log)
diff <(grep "^step" simd_avx512.log) <(grep "^step" simd_off.log)
```

### CUDA back-end (Phase 1, v0.18)

Build with the optional `cuda` feature (requires the CUDA toolkit on
the host - on Arch Linux that is `sudo pacman -S cuda`, which puts
`nvcc` at `/opt/cuda/bin/nvcc` and the libs at `/opt/cuda/lib64`):

```sh
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo build --release --features cuda
```

`cudarc 0.19` uses the `cuda-version-from-build-system` feature, so the
build script reads the toolkit version from `nvcc --version` and you do
not need to pin a specific `cuda-13020` / `cuda-12080` feature in
`Cargo.toml`. `dynamic-loading` finds `libcuda.so` and `libnvrtc.so.*`
at runtime, so the same binary loads on a machine with the toolkit at
a different path.

Phase 1 is matmul-only. The `cuda-demo` subcommand microbenches the
three matmul shapes the v0.13 model leans on hardest:

```sh
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo run --release --features cuda -- cuda-demo
```

Empirical numbers on a 7940HS + RTX 4070 Laptop (driver 595.71.05,
CUDA 13.2, Ada compute capability 8.9):

| Shape                                | CPU AVX-512 | NVRTC tile | cuBLAS sgemm | max abs drift |
|--------------------------------------|------------:|-----------:|-------------:|--------------:|
| Attention Q  `[64,192] @ [192,16]`   |       53 us |      33 us |    60-65 us  |       1.9e-6  |
| FFN gate/up  `[64,192] @ [192,384]`  |      210 us |      97 us |       85 us  |       2.9e-6  |
| FFN down     `[64,384] @ [384,192]`  |      270 us |      91 us |       85 us  |       4.5e-6  |

The NVRTC column is the v0.18 hand-rolled tile-based kernel (kept as a
test-only reference); the cuBLAS column is the v0.18.1 / Chunk 2.0
production path. Surprise sub-result: **cuBLAS loses to the hand-roll
on the smallest shape** (attention Q) because per-call cuBLAS setup -
handle bookkeeping, kernel auto-selection, stream sync - dominates
below ~10k output cells. cuBLAS wins decisively at FFN sizes (24k+
cells), and at Phase 2 scale (one device-side function call per
attention head) the per-call setup amortises across 4-8 invocations
and cuBLAS dominates everywhere.

The CUDA column **includes both H->D copies and the D->H copy back**.
The kernel itself is roughly 5-10 us on a 4070; the rest is PCIe
round-trip. Phase 2 will keep tensors device-resident across whole
blocks so the copy cost is paid once per forward pass rather than once
per matmul - that should push the speedup to roughly 5-10x at v0.13
scale.

The advance prediction in Phase 1 was that GPU would be *slower* than
CPU at v0.13 model size because of copy dominance. That prediction was
wrong: GPU FLOPS scale much faster than copy bandwidth, so even tiny
matmuls win on GPU and the win grows with matmul size. The bigger the
matmul, the bigger the gap.

Cross-path numerical drift is non-zero (the CPU AXPY summation order
and the GPU thread-block summation order differ; f32 add is non-
associative). The `assert_close` helper in `cuda::tests` uses a
tolerance of `1e-4 + 1e-4 * |val|`, an order of magnitude looser than
the actual measured drift (~6e-7 absolute on the test shapes). Resumed
training that switches between CPU and GPU mid-run would lose the
byte-identical-checkpoint guarantee that the CPU SIMD paths have; for
now the training pipeline is CPU-only, so this is a Phase 3 concern.

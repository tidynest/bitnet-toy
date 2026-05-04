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
now the training pipeline is CPU-only, so this is a Phase 4 concern.

### CUDA back-end (Phase 2: trait architecture)

Phase 2 (chunks 2.1-2.4) wraps every op the model needs in a per-op
trait declared in `src/device.rs`. Each trait has a CPU impl on
`Tensor` and a CUDA impl on `CudaTensor`; a single generic helper
function written against the trait surface compiles for both
backends:

```rust
// All in src/device.rs.
pub trait MatMul     { fn matmul(&self, rhs: &Self) -> Self; }
pub trait Add        { fn add(&self, rhs: &Self) -> Self; }
pub trait Mul        { fn mul(&self, rhs: &Self) -> Self; }
pub trait MulScalar  { fn mul_scalar(&self, s: f32) -> Self; }
pub trait Transpose2D{ fn transpose_2d(&self) -> Self; }
pub trait Softmax    { fn softmax(&self) -> Self; }
pub trait CausalMask { fn causal_mask(&self) -> Self; }
pub trait Rope       { fn rope(&self) -> Self; }
pub trait Silu       { fn silu(&self) -> Self; }
pub trait RmsNorm    { fn rmsnorm(&self) -> Self; }

pub fn block_inference<T: MatMul + Add + MulScalar + Transpose2D
                          + Softmax + CausalMask + Rope + RmsNorm
                          + Silu + Mul>(
    x: &T,
    heads: &[HeadWeights<T>],
    ffn: &FfnWeights<T>,
    head_dim: usize,
) -> T;
```

Cross-backend tests (`*_cpu_vs_cuda_matches`) run each generic helper
through `Tensor` and `CudaTensor` over the same input and assert
output agreement within tight FP tolerance. Per-op tolerance is
`1e-4 + 1e-4 * |val|` (actual measured drift ~1-5e-6); chained
attention test passes at 1e-3; chained block test at 5e-3. Failure
of any of these tests implies a regression in a single op kernel.

### CUDA back-end (Phase 3: end-to-end forward)

`CudaModel` (in `src/cuda.rs`) holds device-resident weights mirroring
the CPU `model::Model`. End-to-end forward chains `embed -> blocks
(block_inference) -> rmsnorm -> lm_head matmul`, all device-resident
except the embed lookup which stays CPU-side (the embed table is
small; row-pick + H->D slab is cheaper than maintaining a device-side
lookup).

```sh
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo run --release --features cuda -- cuda-forward-bench
```

Empirical numbers on the same 7940HS + RTX 4070 Laptop:

| Config                                          | CPU (us) | CUDA (us) | Ratio |
|-------------------------------------------------|---------:|----------:|------:|
| tiny (vocab 17, hidden 32, 4 heads, 2 blocks)   |       40 |       922 | 23x slower |
| med  (vocab 65, hidden 96, 6 heads, 4 blocks)   |      538 |      2283 |  4x slower |

**The GPU is far slower at small model scale.** Each forward queues
60-80 kernel launches and each launch pays ~10-30 us of fixed driver
overhead regardless of how cheap the kernel is. At hidden=32 the
kernel itself is ~1 us so launch overhead is ~95% of per-call cost.
v0.13 production scale (hidden 192) gives kernels real work and
should flip the ratio; further wins live in kernel fusion, CUDA
graphs, larger batches, or the future Phase 5 ternary tensor-core
kernels.

Per-op `stream.synchronize()` calls were stripped from every Phase 2
trait impl during Phase 3 - cudarc's `clone_dtoh` (used by `to_cpu`)
is sync, so the final read serialises naturally; queued launches on
the same stream serialise relative to each other without explicit
sync. Stripping cut the med config from ~7x slower to ~4x slower
(the tiny config barely moved because launch overhead, not sync,
dominates there).

### CUDA back-end (Phase 4 chunk 4.1: matmul backward)

The first Phase 4 chunk lands a generic backward helper for matmul on
the existing trait surface, with zero new kernels:

```rust
// src/device.rs
pub fn matmul_backward<T: MatMul + Transpose2D>(
    grad_c: &T, a: &T, b: &T,
) -> (T, T) {
    let grad_a = grad_c.matmul(&b.transpose_2d());
    let grad_b = a.transpose_2d().matmul(grad_c);
    (grad_a, grad_b)
}
```

This is the same closed-form identity the CPU `Var::matmul` backward
closure uses (`autograd.rs:253-260`). Because it composes only
existing traits, both `Tensor` and `CudaTensor` already satisfy the
bounds; no GPU autograd machinery is needed yet.

Two test patterns established here that later Phase 4 chunks (silu,
mul, softmax, rmsnorm, rope) will mirror per op:

1. **CPU finite-difference check.** Pick a small case with non-trivial
   upstream gradient `grad_c`, compute analytic gradients via the
   helper, central-difference check every cell against
   `f(A, B) = sum(grad_c * (A @ B))`. Catches math bugs in the helper
   *and* drift in the underlying ops.
2. **Cross-backend agreement.** Same generic helper monomorphised for
   `Tensor` and `CudaTensor` over identical input bytes; gradient
   tensors agree within `1e-4 + 1e-4 * |val|` for matmul-backward,
   looser tolerances for chained chunks.

### CUDA back-end (Phase 4 chunk 4.2: elementwise backwards)

Chunk 4.2 landed the four elementwise backwards the FFN + residual
paths need. Three are pure trait composition (zero new kernels); the
fourth (`Silu`) gets one fused kernel.

```rust
// src/device.rs
pub fn add_backward<T: Clone>(grad_c: &T) -> (T, T) {
    (grad_c.clone(), grad_c.clone())
}

pub fn mul_backward<T: Mul>(grad_c: &T, a: &T, b: &T) -> (T, T) {
    (grad_c.mul(b), grad_c.mul(a))
}

pub fn mul_scalar_backward<T: MulScalar>(grad_c: &T, s: f32) -> T {
    grad_c.mul_scalar(s)
}

pub trait SiluBackward {
    fn silu_backward(&self, x: &Self) -> Self;   // self = grad_y
}
pub fn silu_backward<T: SiluBackward>(grad_y: &T, x: &T) -> T {
    grad_y.silu_backward(x)
}
```

The `silu_backward_f32` NVRTC kernel reads `grad_y[i]` and `x[i]`,
computes `sig = 1/(1 + exp(-x[i]))` once, writes `grad_y[i] * sig *
(1 + x[i] * (1 - sig))`. Same per-cell formula the CPU
`Var::silu` closure uses. One thread per element; same launch shape
as forward `silu_f32`.

`CudaTensor` is now `Clone` via `clone_dtod` (device-to-device
memcpy). The `add_backward` cross-backend test arm doubles as the
only test of `CudaTensor::clone()`, so a regression in `clone_dtod`
will surface there.

### CUDA back-end (Phase 4 chunk 4.3: softmax + causal-mask backwards)

Chunk 4.3 lands two fused NVRTC kernels for the attention-shape
backwards:

```rust
// src/device.rs
pub trait SoftmaxBackward {
    fn softmax_backward(&self, s_out: &Self) -> Self;  // self = grad_y
}
pub trait CausalMaskBackward {
    fn causal_mask_backward(&self) -> Self;            // self = grad_y
}
pub fn softmax_backward<T: SoftmaxBackward>(grad_y: &T, s_out: &T) -> T;
pub fn causal_mask_backward<T: CausalMaskBackward>(grad_y: &T) -> T;
```

The softmax JVP collapses to one row reduction plus a per-cell
update:

```
dot_i             = sum_k grad_y[i, k] * s_out[i, k]
grad_in[i, j]     = s_out[i, j] * (grad_y[i, j] - dot_i)
```

Critically, `s_out` is the saved softmax **output**, not the input -
the autograd `Var::softmax` saves output for exactly this reason
(autograd.rs:618-621), and the trait method follows suit. Both new
kernels load out of the same NVRTC module as the existing op kernels
(no extra `compile_ptx` cost):

- `softmax_backward_row_f32`: one thread per row matching forward
  `softmax_row_f32`'s launch shape; two passes per row (compute dot,
  then per-cell update).
- `causal_mask_backward_f32`: 2-D 16x16 grid mirroring forward
  `causal_mask_f32`'s shape; one thread per output cell. The kernel
  is purely copy-or-zero, so cross-backend agreement is byte-
  identical, not approximate.

Test pattern note: the causal-mask backward uses a structural check
(identity below diagonal, zero above) rather than finite differences,
because the forward writes `-inf` into the upper triangle, which
would make `sum(grad_y * forward(x))` itself non-finite.

### CUDA back-end (Phase 4 chunk 4.4: rmsnorm + rope backwards)

Chunk 4.4 closes the per-op backward set. After this chunk every
forward op the model uses has a generic backward helper on the trait
surface.

```rust
// src/device.rs
pub trait RmsNormBackward {
    fn rmsnorm_backward(&self, x_saved: &Self) -> Self;  // self = grad_y
}
pub trait RopeBackward {
    fn rope_backward(&self) -> Self;                     // self = grad_y
}
pub fn rmsnorm_backward<T: RmsNormBackward>(grad_y: &T, x_saved: &T) -> T;
pub fn rope_backward<T: RopeBackward>(grad_y: &T) -> T;
```

RMSNorm backward couples every cell of a row through the shared
`rms_i` scalar:

```
inv_rms_i        = 1 / sqrt(mean_j(x_saved[i, j]^2) + EPS)        # EPS = 1e-5
dot_i            = sum_j x_saved[i, j] * grad_y[i, j]
factor_i         = dot_i * inv_rms_i^3 / n
grad_in[i, j]    = grad_y[i, j] * inv_rms_i - x_saved[i, j] * factor_i
```

The kernel recomputes `rms_i` from `x_saved` rather than carrying a
separate `[m]` saved-norms tensor across the trait surface (cost: one
extra row pass; benefit: API stays symmetric with
`softmax_backward(grad_y, s_out)`).

RoPE backward is parameter-free - the angles depend only on shape -
so the trait method is single-arg. Per `(pos, pair)` it applies the
inverse rotation: same trig table as forward with `sin` flipped,
because each per-pair rotation is orthogonal:

```
grad_in[pos, 2i]   =  grad_y[pos, 2i]   * cos + grad_y[pos, 2i+1] * sin
grad_in[pos, 2i+1] = -grad_y[pos, 2i]   * sin + grad_y[pos, 2i+1] * cos
```

Both new kernels (`rmsnorm_backward_row_f32`,
`rope_backward_f32`) load out of the same NVRTC module as the existing
op kernels - no extra `compile_ptx` cost. Launch shapes mirror the
forwards exactly (rmsnorm: one thread per row; rope: 2-D 16x16 over
`(pair, pos)`).

Tolerance note: the cross-backend RoPE test runs at one order of
magnitude looser than the rmsnorm arm (`1e-3` vs `1e-4`) because
`cos`, `sin`, and `powf` differ in the last few mantissa bits between
CPU libm and CUDA's intrinsics, the same trade-off the forward-RoPE
test absorbs.

### CUDA back-end (Phase 4 chunk 4.5.a: attention-head forward+backward)

Chunk 4.5.a wires the chunk-4.1-4.4 op-level backwards into a hand-
traced backward chain for one attention head, expressed against the
trait surface (so it runs on either backend without modification):

```rust
// src/device.rs
pub struct AttentionHeadSaved<T> { pub q: T, pub k: T, pub v: T,
                                   pub attn: T, pub ctx: T }
pub struct AttentionHeadGrads<T>  { pub grad_x: T, pub grad_w_q: T,
                                    pub grad_w_k: T, pub grad_w_v: T,
                                    pub grad_w_o: T }

pub fn attention_head_forward_save<T>(...) -> (T, AttentionHeadSaved<T>)
where T: MatMul + MulScalar + Transpose2D + Softmax + CausalMask + Rope;

pub fn attention_head_backward<T>(grad_out, x, saved, w_q, w_k, w_v, w_o,
                                  head_dim) -> AttentionHeadGrads<T>
where T: MatMul + Add + MulScalar + Transpose2D
       + SoftmaxBackward + CausalMaskBackward + RopeBackward;
```

Two design notes:

- **Saved-tensor selection follows kernel needs.** Matmul backward
  needs both operands; softmax backward needs the saved output;
  RoPE / causal-mask / mul_scalar backwards are shape-only. So the
  Saved struct holds exactly five intermediates - `q`, `k`
  (post-RoPE), `v`, `attn`, `ctx`. Pre-mask / pre-mul_scalar /
  pre-softmax tensors are not saved because their backwards are
  shape-only.
- **No `TransposeBackward` trait.** The matmul `q @ k.T` backward
  gives `grad_q_post` and `grad_kt`; we recover `grad_k_post` by
  transposing `grad_kt` again, since transpose is its own inverse
  for rank-2 tensors. This avoids inflating the trait surface.

The hand-traced chain runs the autograd graph in reverse-creation
order, calling each chunk-4.1-4.4 helper in turn. `grad_x` ends up
as the residual sum of three branches (Q / K / V matmuls all
consume `x`), built via two `Add` calls.

Validation strategy that the rest of chunk 4.5 will mirror:

1. **Autograd ground-truth test.** Build the same head two ways:
   path A via the project's pre-Phase-4 `Var`-based autograd
   (`tape.backward(out.id)` seeds with ones); path B via the new
   helpers on plain `Tensor`. Path B passes `Tensor::ones(out.shape)`
   as `grad_out` for a like-for-like compare. Asserts forward
   equality at `1e-5 + 1e-5 * |val|` and gradient equality at
   `1e-4 + 1e-4 * |val|`. **Strongest correctness gate** - any bug
   in any chunk-4.1-4.4 backward kernel or any wiring mistake in
   the chain shows up here.
2. **Cross-backend test.** Same helper monomorphised for `Tensor`
   and `CudaTensor` over identical inputs. Tolerances loosen to
   `5e-3` forward / `1e-2` backward because the chain composes 5
   matmuls + softmax + 2 ropes + transpose + 3-branch residual sum,
   accumulating drift from cuBLAS / NVRTC parallel-reduction f32
   sums. **First test that exercises every Phase 4 backward kernel
   in one chain on the GPU.**

### CUDA back-end (Phase 4 chunk 4.5.b: SwiGLU FFN forward+backward)

Chunk 4.5.b mirrors the chunk-4.5.a recipe for the FFN:

```rust
// src/device.rs
pub struct FfnSaved<T> { pub gate_pre: T, pub gate: T, pub up: T,
                         pub h: T }
pub struct FfnGrads<T>  { pub grad_x: T, pub grad_w_gate: T,
                          pub grad_w_up: T, pub grad_w_down: T }

pub fn ffn_forward_save<T>(x, w_gate, w_up, w_down) -> (T, FfnSaved<T>)
where T: MatMul + Silu + Mul;

pub fn ffn_backward<T>(grad_y, x, saved, w_gate, w_up, w_down) -> FfnGrads<T>
where T: MatMul + Mul + Add + Transpose2D + SiluBackward;
```

Saved-tensor selection follows the chunk-4.5.a rule:

- `gate_pre` is the input to silu, needed for `silu_backward`.
- `gate` is the silu output, needed for `mul_backward(grad_h, gate, up)`.
- `up` is the second mul operand.
- `h = gate * up` is the input to the final matmul, needed for
  `matmul_backward(grad_y, h, w_down)`.

Trait bounds are notably **shorter** than chunk 4.5.a's set: no
softmax / causal-mask / RoPE / RMSNorm pieces, since the FFN
backward is its own neat island in the trait graph. The cross-
backend test uses **tighter tolerances** than 4.5.a (`1e-3`
forward, `5e-3` backward) because the chain is shorter (3 matmuls
+ silu + mul vs 5 matmuls + softmax + 2 ropes + transpose +
3-branch sum), so less drift accumulates from cuBLAS / NVRTC
parallel-reduction f32 sums.

### CUDA back-end (Phase 5.b: tensor-core int8 GEMM behind BitLinear)

Phase 5.b rewrites the `BitLinear` trait impl on `CudaTensor` to run
the matmul in INT32 on Ada tensor cores via `cublasGemmEx`:

```
y[m, n] = (alpha[m] * gamma / 127) * sum_k x_q[m, k] * w_q[k, n]
```

where `x_q` is INT8 in `[-128, 127]` (per-row scale `alpha[m]`),
`w_q` is INT8 in `{-1, 0, +1}` (scalar scale `gamma`). cuBLAS
arguments: `CUDA_R_8I` for both inputs, `CUDA_R_32I` for the
output, `CUBLAS_COMPUTE_32I` compute type,
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` algo. Same row-major-via-column-
major adapter as the f32 `MatMul` impl.

Three new NVRTC kernels: `quantise_weights_int8_apply_f32`,
`quantise_acts_int8_apply_f32`, `scale_int32_to_f32`. The reduction
kernels (`quantise_weights_ste_partial_abs_sum_f32`,
`quantise_acts_ste_row_absmax_f32`) are reused from Phase 5.a; only
the apply kernels (which write INT8 instead of dequantised f32) and
the final scale kernel are new.

**Shape fallback**: `cublasGemmEx` int8 GEMM requires `lda` and
`ldb` to be multiples of 4 (the int8 kernel reads 32-bit chunks).
In our adapter that means `n` and `k` must both be multiples of 4.
The only project matmul that violates this is lm_head with
`n = vocab = 65`; the `BitLinear` impl checks alignment up front
and falls through to the Phase 5.a f32 path when it fails. Both
paths are algebraically identical so output values match within
f32 round-off.

**Empirical reality check** on the 7940HS + RTX 4070 Laptop at
v0.13 scale (~5M params): Phase 5.b cuda-shakespeare runs at
~300 ms/step; CPU shakespeare is ~180 ms/step. **GPU is currently
slower than CPU** because per-step launch overhead dominates -
each training step does ~3000+ kernel launches (~50 bit_linear
calls per window x 4 windows in batch x 6 quant + GEMM + scale
launches per bit_linear x 2 for forward + backward). At 10-30 us
launch overhead per kernel that is 30-90 ms of pure overhead per
step; the int8 GEMM kernel runs in microseconds on tensor cores
so the actual matmul work is dwarfed.

Three orthogonal post-Phase-5.b optimisations to realise the
tensor-core speedup:
- **Kernel fusion**: collapse the 4 quant kernels per bit_linear
  (per-row reduction + per-cell apply, for both x and w) into 1-2
  launches.
- **CUDA graphs**: capture a full training-step launch sequence
  once and replay it. cudarc 0.19 has graph support.
- **Larger batches / longer sequences**: amortise launch overhead
  across more matmul work. `shakespeare-large` (~8.5M params,
  seq_len 128) is the obvious test target.

### CUDA back-end (Phase 5.a: real BitNet ternary GPU training)

Phase 5.a wraps the chunk-4.5 GPU training pipeline with BitNet b1.58
STE quantisation, producing real ternary checkpoints on the GPU.

```rust
// src/device.rs
pub trait QuantiseWeightsSTE { fn quantise_weights_ste(&self) -> Self; }
pub trait QuantiseActsSTE    { fn quantise_acts_ste(&self) -> Self;   }
pub trait BitLinear          { fn bit_linear(&self, rhs: &Self) -> Self; }
pub fn bit_linear_backward<T: ...>(grad_y: &T, x: &T, w: &T) -> (T, T);
```

`bit_linear(x, w) = quantise_acts_ste(x) @ quantise_weights_ste(w)`.
STE makes both quants identity in backward, so `bit_linear_backward`
just re-quantises x and w on the fly and calls regular
`matmul_backward`. Re-quantising during backward is cheaper than
carrying extra saved tensors across the trait surface.

Four new NVRTC kernels (per-row reduction + per-cell apply for both
weights and activations). Edge case: a row of all zeros has alpha = 0;
the act-apply kernel writes zero directly to avoid NaN.

`_bitnet` variants of every chunk-4.5.x layer helper (`attention_head_*`,
`ffn_*`, `block_*`) swap learnable-weight matmuls for `bit_linear`;
internal act-against-act matmuls (Q @ K.T, attn @ V) stay unquantised
- matches the autograd `attention::head_output` recipe in
`attention.rs` byte-for-byte.

`CudaModel::compute_grads_for_window_bitnet` is the bitnet sibling of
chunk-4.5.e's `compute_grads_for_window`. New
`compute_batched_grads_cuda_bitnet` in `main.rs` mirrors the existing
`compute_batched_grads`'s averaged-batch behaviour; new
`TrainConfig.use_cuda_backward` flag dispatches `train_bitnet_lm`
through it. New CLI subcommands:

```sh
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo run --release --features cuda -- cuda-shakespeare
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo run --release --features cuda -- cuda-shakespeare-large
PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo run --release --features cuda -- cuda-shakespeare \
    models/shakespeare.v0.13.30k-resumed.f32.bin
```

The CUDA training subcommand reuses the entire existing CPU
training loop - val_ppl tracking, AdamW, lossless `.f32.bin` +
compact `.ternary_packed.bin` export, four sampling-mode generation
passes, resume support sharing the same on-disk format. Output
checkpoints are real BitNet 1.58-bit-per-weight artefacts.

Per-step cost decomposition (Phase 5.a, v0.13 ~5M-param config):
- `CudaModel::from_cpu(&model)` rebuild: ~1 ms (H->D copy of every
  weight tensor + allocation). Done once per training step. Removable
  via `CudaModel::sync_from_cpu(&Model)` (queued).
- Quant kernels (per learnable matmul): per-row reduction + per-cell
  apply. Microsecond range each.
- cuBLAS sgemm: still f32 internally (Phase 5.b will replace with
  tensor-core int8 GEMM for the ~10-50x speedup).
- Backward chain: regular `matmul_backward` (cuBLAS sgemm again) +
  the chunk-4.1-4.4 backward kernels.
- AdamW step on CPU: linear in param count.

Headline correctness gate:
`attention_head_backward_bitnet_matches_autograd_ground_truth` builds
the same head two ways - autograd `Var` with explicit
`quantise_acts_ste`/`quantise_weights_ste` wrappers (matching
`attention::head_output` byte-for-byte) vs the chunk-5.a helpers -
and asserts every gradient cell agrees at `1e-4`.

### CUDA back-end (Phase 4 chunks 4.5.c-f: end-to-end GPU training)

Chunks 4.5.c through 4.5.f compose the per-op and per-layer
backwards into a runnable training step:

- **4.5.c `block_forward_save<T>` + `block_backward<T>`**: full
  pre-norm transformer block. Composes attention + FFN + 2 RMSNorms
  + 2 residual `Add` backwards. The Add backwards are identity
  (Clone); each residual splits its incoming gradient along both
  branches. Multi-head sum-of-projections means each head's
  `attention_head_backward` receives the same upstream gradient and
  per-head input gradients accumulate.
- **4.5.d `CrossEntropy` + `CrossEntropyBackward` traits**: fused
  softmax + per-row loss in `cross_entropy_forward_save`; closed-form
  `(softmax - onehot) / seq` gradient in `cross_entropy_backward`.
  Two new NVRTC kernels (`cross_entropy_softmax_loss_f32` for
  forward; `cross_entropy_backward_f32` for backward).
- **4.5.e `CudaModel::compute_grads_for_window(input, target)
  -> (Vec<Tensor>, f32)`**: the integration. Embed gather (CPU) +
  H->D + chained `block_forward_save` + final RMSNorm + lm_head
  matmul -> logits -> cross-entropy fwd. Backward runs the chain
  in reverse, scatters the embed-input gradient into a CPU
  `grad_token_embed` table, and flattens every device-side gradient
  to CPU in canonical visitor order. Drop-in compatible with the
  CPU `compute_grads_for_window` signature.
- **4.5.f `cuda-train-demo` CLI subcommand**:
  ```
  cargo run --release --features cuda -- cuda-train-demo
  ```
  Builds a tiny model from `TINY_CORPUS`, runs 100 training steps
  on GPU + AdamW updates on CPU, prints loss trajectory. Empirical:
  loss 3.0015 -> 1.3106 in 100 steps, 0.4 s wall-clock on the
  7940HS + RTX 4070 Laptop.

The path is **f32 throughout** - BitNet ternary STE quantisation is
not applied. Phase 5 (tensor-core ternary kernels) restores BitNet
semantics on the GPU. Until then, the GPU training path trains an
f32 transformer (which is still useful for ablations + perf
measurements + Phase 4 wiring proof) but is not a drop-in
replacement for the CPU path's BitNet ternary semantics.

Per-step cost decomposition (v0.13 demo scale):
- `CudaModel::from_cpu(&model)` rebuild: ~1 ms (H->D copy of every
  weight tensor + allocation). Done once per training step because
  CPU weights mutate after each AdamW update. Removable in a
  follow-up via `CudaModel::sync_from_cpu(&Model)` that overwrites
  existing device buffers in place.
- Forward + backward through one block stack: tens of kernel
  launches at ~10-30 us each. Dominated by launch overhead at this
  model scale; production-scale kernels (v0.13 hidden=192) would
  shift to compute-bound.
- AdamW step on CPU: linear in param count; ms range.

### CUDA back-end (preflight)

Phase 3 surfaced a class of bugs that the silent-skip pattern in
GPU tests (`if cuda_state().is_err() { return }`) would otherwise
hide for many chunks: an NVRTC kernel that fails to compile makes
`cuda_state()` return `Err`, every test silently skips as `ok`, and
no GPU code is exercised. Defence: a hard preflight test gated by
`EXPECT_CUDA=1`. On a machine that has a usable GPU, always run with
the env var set so any NVRTC regression panics immediately:

```sh
EXPECT_CUDA=1 PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \
    cargo test --release --features cuda \
    cuda::tests::preflight_cuda_state_initialises_cleanly_when_expected
```

This costs nothing on green builds and catches the silent-skip
failure mode immediately on any future kernel-compile breakage.

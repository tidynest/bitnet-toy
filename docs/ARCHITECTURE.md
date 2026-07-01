# Architecture

![docs](https://img.shields.io/badge/docs-architecture-3b6ea5)
![version](https://img.shields.io/badge/version-v0.19.0-3b6ea5)

How the modules compose, top-down.

## Contents

- [Layered view](#layered-view)
- [Why this layering](#why-this-layering)
- [The training cycle](#the-training-cycle-one-step)
- [STE in one paragraph](#ste-in-one-paragraph)
- [Multi-head attention](#multi-head-attention-sum-of-projections)
- [SwiGLU FFN](#swiglu-ffn)
- [Causal mask](#causal-mask)
- [Per-op trait architecture (Phase 2)](#per-op-trait-architecture-phase-2)
- [CUDA back-end](#cuda-back-end)
- [Memory model](#memory-model)
- [Where the toy stops](#where-the-toy-stops)

## Layered view

```
                                 main.rs
                  (TrainConfig, CLI: shakespeare /
                   shakespeare-large / sample / cuda-demo /
                   cuda-forward-bench / cuda-train-demo /
                   cuda-shakespeare[-large], demos, tests)
                                    |
       ┌────────────────────────────┼────────────────────────────┐
       |                            |                            |
  inference.rs            inference_kv.rs                   optim.rs
 (greedy + temp +    (KV-cached generation,             (AdamW, clip,
  top-k + top-p)      ~50-100x faster                    cosine LR
                      per-token vs full forward)         + warmup)
                            |                                |
                            └─────────┬──────────────────────┘
                                      |
                                  model.rs
                  (Model, ModelConfig, leaf register, init)
                                      |
                ┌─────────────────────┼─────────────────────┐
                |                     |                     |
             block.rs            attention.rs            ffn.rs
        (RMSNorm + attn       (multi-head, causal,    (SwiGLU: gate via
         + FFN + residual)    RoPE, sum-of-           SiLU, gated up,
                              projections)            down projection)
                                      |
                                autograd.rs
            (Tape, Var, all ops: matmul, softmax, rmsnorm,
             rope, silu, quantise_*_ste, causal_mask, embed,
             cross_entropy, etc.)
                                      |
              ┌───────────────────────┼───────────────────────┐
              |                       |                       |
         tensor.rs              device.rs               bitlinear.rs
     (raw f32 storage,    (per-op traits: MatMul,    (absmean_ternary,
      AVX-512/AVX2/scalar  Add, Mul, MulScalar,       absmax_int8)
      matmul, parallel     Transpose2D, Softmax,
      via thread::scope)   CausalMask, Rope, Silu,
                           RmsNorm; generic helpers
                           attention_head_inference,
                           ffn_inference, block_inference)
                                      |
                                   cuda.rs                       export.rs
                       (CudaContext + cuBLAS handle +     (binary I/O,
                        NVRTC kernels for every op       three formats,
                        in device.rs; CudaTensor;        round-trip
                        CudaModel + end-to-end forward;  importer; OPTM
                        cuda-demo + cuda-forward-bench)  payload for
                        Gated behind --features cuda     resume)

                              data.rs
              (Vocab, sliding windows, LCG, file reader; standalone)
```

The `device.rs` traits are the abstraction boundary that lets the same
generic helper function (`block_inference<T>`, `ffn_inference<T>`,
`attention_head_inference<T>`) compile and run on both `Tensor` (CPU)
and `CudaTensor` (GPU) - validated by cross-backend tests that demand
output agreement within tight FP tolerance on every shared call site.

## Why this layering

- **`tensor.rs`** is a pure value type. No autograd awareness, no quantisation.
  Pure linear algebra primitives. Holds the matmul kernel which
  dispatches at runtime to AVX-512, AVX2, or scalar (per
  `matmul_simd_mode`); also the per-row sharded parallel matmul
  driven by `BITNET_MATMUL_THREADS`. All three SIMD widths are
  bit-identical per output cell because none use FMA.

- **`device.rs`** is the device-abstraction layer (Phase 2). One
  trait per op family (`MatMul`, `Add`, `Mul`, `MulScalar`,
  `Transpose2D`, `Softmax`, `CausalMask`, `Rope`, `Silu`, `RmsNorm`),
  each implemented on `Tensor` (CPU) and `CudaTensor` (GPU). Plus
  generic-over-backend helpers (`attention_head_inference<T>`,
  `ffn_inference<T>`, `block_inference<T>`) that compose those
  traits into model layers.

- **`cuda.rs`** is the CUDA back-end (gated `#[cfg(feature = "cuda")]`).
  Holds the NVRTC-compiled kernels for every op in `device.rs`, the
  cuBLAS handle for matmul, `CudaTensor` (device-resident f32
  storage), and `CudaModel` (end-to-end forward). The kernels live in
  one `KERNELS_SRC` const string; `cuda_state()` compiles them once
  per process and caches function handles. Production matmul uses
  cuBLAS sgemm; the v0.18 hand-rolled tile-GEMM kernel is kept
  `#[cfg(test)]` as an independent reference for cross-checking.

- **`bitlinear.rs`** holds the two quantisation primitives as free functions.
  These are the only places that compute gamma and alpha; everything else
  consumes `(W_q, gamma)` or `(x_q, alpha)` returned from here.

- **`autograd.rs`** is the centerpiece. It owns the `Tape` data structure,
  the `Var` handle, and every op the project needs. STE wrappers
  (`quantise_weights_ste`, `quantise_acts_ste`) live here too, so the
  autograd-aware path stays in one file.

- **`attention.rs`, `ffn.rs`, `block.rs`** are short composition layers built
  on `Var` ops. They contain no state; they're pure functions taking `Var`s
  in and returning a `Var`.

- **`model.rs`** owns the master parameters as `Tensor`s and exposes a
  parameter-visitor API. The `register_leaves` method binds a fresh
  `Var` set to a tape for one forward; `apply_grads` and `for_each_param_with_grad`
  let optimisers iterate the model's tensors uniformly.

- **`optim.rs`** consumes the visitor. AdamW maintains its own `m`, `v`
  buffers indexed by visitor order. Gradient clipping touches the leaves
  directly via `Tape::write_grad`.

- **`export.rs`** owns the on-disk format. Header + payload.
  Round-trippable in all three formats. The format byte plus a 4-byte magic
  let importers detect the right reader.

- **`inference.rs`** is the original autograd-path generator: greedy +
  temperature + top-k + top-p sampling. Builds the full Var forward
  graph each step (slow but matches training-time behaviour
  exactly).

- **`inference_kv.rs`** (v0.16; sliding window since v0.16.1) is the
  KV-cached generator: per-block per-head K and V tensors grow by one
  row per step instead of being recomputed, capped at `max_seq_len`
  rows (oldest row evicted when full). K is stored *unrotated*; RoPE
  is applied at attention-score time using each row's logical
  (in-cache) position so positions stay in the trained `[0,
  max_seq_len - 1]` range for arbitrarily long generations. ~50-100x
  faster per-token generation. Pure `Tensor` + `Vec<f32>` math, no
  autograd, no tape. Shape parity with the autograd path is asserted
  by `cached_forward_matches_var_forward_to_within_floating_point_drift`
  (no-eviction regime); the sliding regime is gated by
  `kv_cache_caps_at_max_seq_len_when_more_tokens_arrive` and
  `cached_forward_stays_in_distribution_past_max_seq_len`.

- **`data.rs`** stands alone. Vocab, sliding windows, file reader, LCG,
  shuffler. No upstream dependencies on anything except `std`.

- **`main.rs`** is the integration layer: CLI dispatch, `TrainConfig` struct
  with its hyperparameters, the M4-M10 demos, and the integration tests
  that gate each milestone.

## The training cycle (one step)

```
master tensors live in Model (outside any tape)
    │
    ▼
Model::register_leaves(&tape)
    Var leaves on the tape, one per master tensor
    │
    ▼
Model::forward(&leaves, ids)
    builds the graph: embed → blocks → final RMSNorm → tied-embedding LM-head matmul
    every op records itself on the tape
    │
    ▼
logits.cross_entropy(targets)
    fused softmax + NLL with closed-form (softmax − onehot) backward
    │
    ▼
tape.backward(loss.id)
    seeds output grad with ones, walks tape in reverse
    each leaf accumulates ∂L/∂master in its grad cell
    │
    ▼
clip_grad_norm(&leaves, 1.0)
    rescales every leaf grad in place if global L2 norm exceeds the cap
    │
    ▼
opt.step(&mut model, &leaves)
    visits every (master, grad) pair and applies AdamW update
    │
    ▼
tape drops at end of scope
    every Var, every saved tensor, every backward closure released
    only the master Tensors persist into the next step
```

## STE in one paragraph

Forward through `quantise_weights_ste(w)` returns `gamma * W_q` where
`(W_q, gamma) = absmean_ternary(w)`. The forward value passes through ternary
quantisation. The backward closure ignores the discreteness and returns the
incoming gradient unchanged: `vec![(parent_id, grad.clone())]`. This is the
"straight-through" lie. Empirically it works because gradient *direction*
matters more than its exact magnitude under SGD/AdamW, and the quantiser
preserves direction information (the master moves; the ternary form follows).

## Multi-head attention (sum-of-projections)

`attention.rs` runs `n_heads` independent attention paths and sums their
outputs. Each head holds its own Q/K/V/O ternary projections in
`AttentionHead` (master tensors) and `AttentionHeadVars<'t>` (tape leaves).
Per-head shapes:

```
W_q, W_k, W_v : [hidden_dim, head_dim]
W_o           : [head_dim,  hidden_dim]
```

Sum-of-projections is mathematically identical to the canonical "concat
heads then project once with a wide W_o" form, because matrix multiplication
distributes over horizontal block concatenation:

```
[H_1 | H_2 | ... | H_n] · [W_o_1; W_o_2; ...; W_o_n]
                = sum over i of H_i · W_o_i
```

The sum form avoids needing a `concat` operation in the autograd. The 1/√d_k
scaling and the causal mask are per-head, applied to each head's own scores
before softmax. The input `x` is INT8-quantised once and reused across all
3 * n_heads projection paths; the tape's per-cell gradient accumulator
gathers gradient from every path back into a single `x` gradient cell.

Conventional sizing: `n_heads * head_dim == hidden_dim`. With this
invariant the total attention parameter count is identical to a single-head
model with `head_dim = hidden_dim`, but the representation is split across
`n_heads` orthogonal subspaces.

## SwiGLU FFN

The position-wise feed-forward network uses the SwiGLU form (LLaMA, BitNet
b1.58). Three weight matrices per block, all BitLinears:

```
gate = silu(x · W_gate)        # [seq, ffn_dim]
up   =       x · W_up           # [seq, ffn_dim]
h    = gate ⊙ up                # element-wise gated product
y    =        h · W_down        # [seq, hidden_dim]
```

`silu(x) = x · σ(x)` where `σ(x) = 1 / (1 + exp(-x))`. Smooth alternative to
ReLU; differentiable everywhere; small leaky negative response avoids the
dead-neuron problem.

Why SwiGLU rather than ReLU:

- The element-wise product `gate ⊙ up` is per-channel gating: the gate
  decides per position and per feature how much of the up-projection to
  pass through. This lets the FFN learn to suppress irrelevant features at
  fine granularity rather than relying on the down projection alone.
- SiLU on the gate keeps the activation continuous, which makes the gating
  decision smooth rather than a hard ReLU cutoff.
- The up projection is left linear so all the non-linearity lives in the
  gate. This is the GLU lineage: linear * non-linear, learned end-to-end.
- Empirically SwiGLU outperforms ReLU and GELU at equal parameter count
  in language modelling (Shazeer 2020, also confirmed in the LLaMA
  ablations).

The cost is one extra weight matrix (W_gate) per block. For BitNet that
extra weight is also ternary, so the on-disk overhead is just one more
4-byte gamma + (hidden_dim * ffn_dim) i8 entries per block.

`Var::silu` lives in `autograd.rs` next to `Var::relu`; backward computes
`d/dx[silu] = σ(x) · (1 + x · (1 − σ(x)))` from the saved input.

## Causal mask

Without causal masking, attention at position `i` could see input row `i+1`,
trivially predicting the target `input[i+1]`. The model would converge to
near-zero loss without learning language. `Var::causal_mask` sets
`scores[i, j] = -inf` for `j > i` before softmax, so positions to the right
get zero attention weight. Backward zeroes gradient on the upper triangle and
passes it through on the lower, matching the forward selection.

## Per-op trait architecture (Phase 2)

Goal: write a model layer once, run it on either CPU or GPU. The
constraint: incremental rollout, so we cannot block on implementing
every op in one trait. The shape: one trait per op family.

```rust
pub trait MatMul    { fn matmul(&self, rhs: &Self) -> Self; }
pub trait Add       { fn add(&self, rhs: &Self) -> Self; }
pub trait Mul       { fn mul(&self, rhs: &Self) -> Self; }
pub trait MulScalar { fn mul_scalar(&self, s: f32) -> Self; }
pub trait Transpose2D { fn transpose_2d(&self) -> Self; }
pub trait Softmax   { fn softmax(&self) -> Self; }
pub trait CausalMask{ fn causal_mask(&self) -> Self; }
pub trait Rope      { fn rope(&self) -> Self; }
pub trait Silu      { fn silu(&self) -> Self; }
pub trait RmsNorm   { fn rmsnorm(&self) -> Self; }
```

All return `Self` (panic on backend errors) so generic helpers compose
without `?` / `.expect()` noise. CPU side panics on shape mismatch;
GPU side `.expect()`s cudarc errors at the trait-impl boundary. Same
panic-on-invariant model on both sides.

Generic helpers (also in `device.rs`):

```rust
fn attention_head_inference<T: MatMul + MulScalar + Transpose2D
                              + Softmax + CausalMask + Rope>(
    x: &T, w_q: &T, w_k: &T, w_v: &T, w_o: &T, head_dim: usize) -> T

fn multi_head_attention_inference<T: ...>(
    x: &T, heads: &[HeadWeights<T>], head_dim: usize) -> T

fn ffn_inference<T: MatMul + Silu + Mul>(
    x: &T, w_gate: &T, w_up: &T, w_down: &T) -> T

fn block_inference<T: MatMul + Add + MulScalar + Transpose2D
                      + Softmax + CausalMask + Rope + RmsNorm
                      + Silu + Mul>(
    x: &T, heads: &[HeadWeights<T>], ffn: &FfnWeights<T>,
    head_dim: usize) -> T
```

The trait-bound list grows linearly with op count (one bound per new
op a helper uses), not exponentially. Cross-backend tests assert that
running each generic helper through `Tensor` and through `CudaTensor`
produces matching outputs within FP tolerance - 1e-3 for
`attention_head_inference`, 5e-3 for `block_inference`, 2e-2 for the
end-to-end `CudaModel::forward`. The per-op tolerances are tighter
(~1e-4 absolute, often much less in practice).

This is **forward-only** for the existing helpers: they do not
interact with autograd. The `Var`-based CPU forward path in
`attention.rs`, `ffn.rs`, and `block.rs` still drives training. Phase
4 is incrementally adding generic backward helpers on the same trait
surface.

Chunk 4.1 landed `matmul_backward<T: MatMul + Transpose2D>(grad_c, a,
b) -> (grad_a, grad_b)` with zero new kernels - pure composition of
existing matmul + transpose ops. Chunk 4.2 added `add_backward<T:
Clone>`, `mul_backward<T: Mul>`, `mul_scalar_backward<T: MulScalar>`
(all kernel-free) and a new `SiluBackward` trait with a fused
`silu_backward_f32` NVRTC kernel. Chunk 4.3 added `SoftmaxBackward`
(per-row JVP `dot_i + cell-wise update`) and `CausalMaskBackward`
(lower-triangle pass-through), both with new fused kernels. Chunk
4.4 added `RmsNormBackward` (per-row coupled JVP through the
`dot * inv_rms^3 / n` factor) and `RopeBackward` (inverse rotation
with `sin` flipped), again both with new fused kernels. Chunk
4.5.a then composed all of the above into
`attention_head_forward_save<T>` + `attention_head_backward<T>` -
the first hand-traced backward chain on the trait surface, validated
against `Var`-based autograd as ground truth at `1e-4` per cell and
across backends at `1e-2`. Chunk 4.5.b mirrored the recipe for the
SwiGLU FFN with `ffn_forward_save<T>` + `ffn_backward<T>`. Chunk 4.5.c
composed both into `block_forward_save<T>` + `block_backward<T>` (one
full pre-norm transformer block). Chunk 4.5.d added cross-entropy
forward+backward (fused softmax + log-sum-exp loss + the closed-form
`(softmax - onehot) / seq` gradient). Chunk 4.5.e tied everything
together in `CudaModel::compute_grads_for_window(input, target) -> (Vec<Tensor>, f32)`
matching the CPU `compute_grads_for_window` signature exactly. Chunk
4.5.f added a `cuda-train-demo` CLI subcommand that runs 100 training
steps end-to-end on GPU (loss 3.0 -> 1.3, ~4 ms/step on the 7940HS +
RTX 4070 Laptop). **Phase 4 is functionally complete.** The path is
f32 throughout; Phase 5 (tensor-core ternary kernels) restores BitNet
semantics on the GPU.

## CUDA back-end

Phases 1-3 (matmul, per-op traits, end-to-end forward), Phase 4 (full GPU
autograd, chunks 4.1-4.5.f), and Phase 5.a/5.b (real ternary BitNet training on
int8 tensor cores) — all optional, gated behind `--features cuda`. Default
`cargo build` stays
dependency-free; CUDA work uses `cudarc 0.19` with `dynamic-loading`
so the same binary loads on machines with the toolkit at non-default
paths.

`cuda_state()` is a process-level `OnceLock`-cached holder that
attaches to GPU 0, instantiates a cuBLAS handle on the default
stream, and NVRTC-compiles `KERNELS_SRC` (one big string holding
every op kernel). Every Phase 2 trait impl on `CudaTensor` launches
its kernel from this cached holder.

`CudaModel` (Phase 3) mirrors `model::Model` but holds device-resident
weights built on the generic `HeadWeights<CudaTensor>` /
`FfnWeights<CudaTensor>` types. Token embedding stays on CPU (small
table; row-pick + H->D slab is cheaper than maintaining a device-side
lookup). `CudaModel::forward(ids) -> CudaTensor` chains
`embed -> blocks(block_inference) -> rmsnorm -> lm_head matmul`, all
device-resident.

Production matmul is **cuBLAS sgemm** with the row-major-via-column-
major adapter (pass B and A in swapped order; cuBLAS sees row-major
as transposed col-major; no transpose flags needed). The v0.18
hand-rolled tile-GEMM kernel is retained `#[cfg(test)]` as an
independent reference that the cuBLAS path is cross-checked against.

**Trap discovered + defended against in Phase 3.** `INFINITY` is not
defined under NVRTC (no host `<math.h>` is auto-included). The first
implementation of `causal_mask_f32` and `softmax_row_f32` used it
freely; NVRTC compile failed and `cuda_state()` returned `Err`,
which the `if cuda_state().is_err() { return }` early-exit pattern in
every CUDA test treated as "skip" - so for several chunks the cross-
backend tests reported `ok` while never actually exercising any GPU
code. Fix: `#define INFINITY (__int_as_float(0x7f800000))` at the top
of `KERNELS_SRC`. Defence: `preflight_cuda_state_initialises_cleanly_when_expected`
test that, when `EXPECT_CUDA=1` is exported, panics if `cuda_state()`
errors for any reason. Catches NVRTC-compile regressions before the
test harness can hide them.

**Performance note.** GPU forward at small model scale is
**slower than CPU** because each forward queues 60-80 kernel launches
and each launch pays ~10-30 us of fixed driver overhead - dominant
when the kernel itself is microseconds. v0.13 production scale
(hidden 192) gives kernels real work and should flip the ratio;
further wins live in kernel fusion, CUDA graphs (capture-and-replay),
larger batches, or Phase 5 ternary tensor cores. Per-op `synchronize()`
calls were stripped from every trait impl during Phase 3 - cudarc's
`clone_dtoh` (used by `to_cpu`) is sync, so the final read serialises
naturally; queued launches on the same stream serialise relative to
each other without explicit sync.

## Memory model

The `Tape` owns every recorded value and gradient cell. `Var<'t>` is a `Copy`
handle into a tape; cloning a `Var` is free, the underlying storage is in the
tape's `Vec<Node>`. Building one tape per training step and dropping it after
backward gives bounded memory: the maximum is one step's worth of saved
forward tensors plus their grad cells.

Master parameters in `Model` live across steps; everything else is ephemeral.

## Where the toy stops

What now works on the GPU (no longer gaps): **end-to-end GPU training**
(Phase 4) and **real ternary BitNet on int8 tensor cores** via `cublasGemmEx`
(Phase 5.a/5.b). `cuda-shakespeare` trains genuine 1.58-bit-per-weight
checkpoints on the GPU.

The places this implementation still diverges from production BitNet, in order
of how much each matters (tracked as
[GitHub issues](https://github.com/tidynest/bitnet-toy/issues)):

1. **GPU is slower than CPU at this scale.** The ternary int8 GEMM is correct
   and runs on Ada tensor cores, but each training step issues ~3000+ kernel
   launches and per-launch driver overhead dominates the microsecond GEMMs.
   Realising the speedup needs device-buffer reuse (`sync_from_cpu`),
   quant-kernel fusion, CUDA graphs, and larger batches — milestone
   **Phase 5.c — GPU perf** (issues #1-#4).

2. **f32 master weights.** Real BitNet keeps BF16 masters; the GEMM is now int8
   on tensor cores, but the master weights and the AdamW optimiser state remain
   f32. Lowering the masters to BF16 is a possible future step, not yet
   scheduled.

3. **AVX-512 underperforms AVX2 on Zen 4.** The 16-wide AVX-512 loads cause more
   L2/L3 contention across 4 worker threads than the 8-wide AVX2 loads on this
   hardware. The auto SIMD selection picks AVX-512 by default; export
   `BITNET_MATMUL_SIMD=avx2` on Zen 4. A CPUID-based auto-select is issue #5.

4. **Inference KV cache is CPU-only.** `inference_kv.rs` is fast (~50-100x over
   the autograd path) but not yet wired through `CudaModel`.

The GPU-perf and SIMD items are tracked on the
[roadmap board](https://github.com/users/tidynest/projects/7).

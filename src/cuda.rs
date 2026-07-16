//! CUDA back-end (Phases 1 + 2.0: matmul-only, cuBLAS sgemm).
//!
//! Production matmul path is **cuBLAS sgemm** (Chunk 2.0). The v0.18
//! hand-rolled NVRTC tile-based GEMM is retained `#[cfg(test)]` as an
//! independent reference implementation that the cuBLAS path is
//! cross-checked against. **NOT bit-identical** to CPU (parallel
//! reduction across thread blocks / SMs reorders the per-cell sum,
//! which is non-associative in f32). Agreement is within roughly
//! `1e-4 + 1e-4 * |val|` for the tensor magnitudes the model produces;
//! empirically the actual drift is ~1-5e-6 absolute on Phase 1 / 2.0
//! test shapes.
//!
//! Phase 1 surface (this commit):
//!   - `CudaTensor` owns one `CudaSlice<f32>` plus row-major shape
//!   - `CudaTensor::from_cpu(&Tensor)` / `to_cpu()` for explicit H<->D
//!   - `CudaTensor::matmul(&Self) -> Self` runs entirely device-side
//!   - `cuda_matmul(&Tensor, &Tensor)` convenience that copies in,
//!     multiplies, copies back (slow per call; useful for tests + demos)
//!
//! Phase 2+ (future sessions):
//!   - `CudaTensor`-resident attention head, then full forward, then
//!     backward, then optimiser - so weights / activations / gradients
//!     live on device across the entire training step.
//!   - Quant-aware kernels: skip the f32 multiply when the weight is
//!     +1 / -1 / 0; this is where BitNet's tensor-core advantage lives.
//!
//! The whole module is gated behind `#[cfg(feature = "cuda")]` so the
//! default `cargo build` stays dep-free and CI on machines without the
//! CUDA toolkit still passes. Run the GPU tests with
//! `cargo test --release --features cuda`.

use std::sync::{Arc, OnceLock};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::tensor::Tensor;

/// Concatenated CUDA C source for the production op kernels (add,
/// mul_scalar, transpose_2d, causal_mask, softmax, rope, silu, mul,
/// rmsnorm). Compiled once per process via NVRTC and cached in
/// `CudaContextHolder`. Each kernel is small (5-25 lines) and uses
/// only `__global__` entry points (no device-only helpers, no template
/// kernels), so the compile is fast and the resulting PTX is human-
/// inspectable if anything misbehaves.
///
/// `expf`, `cosf`, `sinf`, `powf`, `sqrtf` come from the CUDA math
/// library and are available without an explicit include in NVRTC.
/// `INFINITY` is **not** defined by NVRTC out of the box (no host
/// `<math.h>` is auto-included), so we provide our own using the
/// `__int_as_float` built-in intrinsic that maps the IEEE-754 +infinity
/// bit pattern (`0x7f800000`) to an f32. This is the same value the
/// host header would have produced.
const KERNELS_SRC: &str = r#"
#define INFINITY (__int_as_float(0x7f800000))

extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

extern "C" __global__ void mul_scalar_f32(
    const float* __restrict__ x,
    float s,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] * s;
}

extern "C" __global__ void transpose_2d_f32(
    const float* __restrict__ in_,
    float* __restrict__ out,
    int r, int c)
{
    // Thread (b, a) writes out[a, b] = in_[b, a]. Output shape [c, r].
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a < c && b < r) {
        out[a * r + b] = in_[b * c + a];
    }
}

extern "C" __global__ void causal_mask_f32(
    const float* __restrict__ in_,
    float* __restrict__ out,
    int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        float v = in_[i * n + j];
        out[i * n + j] = (j > i) ? -INFINITY : v;
    }
}

extern "C" __global__ void softmax_row_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int m, int n)
{
    // One thread per row. Three passes (max / sum-of-exps / normalise),
    // matching the CPU code exactly. Trades intra-row parallelism for
    // simplicity; at our model scale (n = seq_len <= 128) the per-row
    // cost is small enough that this is not the bottleneck.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float* xr = x + row * n;
    float* yr = out + row * n;

    float row_max = -INFINITY;
    for (int j = 0; j < n; ++j) {
        float v = xr[j];
        if (v > row_max) row_max = v;
    }
    float denom = 0.0f;
    for (int j = 0; j < n; ++j) {
        float e = expf(xr[j] - row_max);
        yr[j] = e;
        denom += e;
    }
    float inv = 1.0f / denom;
    for (int j = 0; j < n; ++j) yr[j] *= inv;
}

extern "C" __global__ void rope_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int seq, int head_dim)
{
    // One thread per (pos, pair). pair indexes 0..head_dim/2.
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pos  = blockIdx.y * blockDim.y + threadIdx.y;
    int half = head_dim / 2;
    if (pair >= half || pos >= seq) return;

    float theta_i = powf(10000.0f, -(2.0f * (float)pair) / (float)head_dim);
    float angle   = (float)pos * theta_i;
    float c = cosf(angle);
    float s = sinf(angle);
    float a = x[pos * head_dim + 2 * pair];
    float b = x[pos * head_dim + 2 * pair + 1];
    out[pos * head_dim + 2 * pair]     = a * c - b * s;
    out[pos * head_dim + 2 * pair + 1] = a * s + b * c;
}

extern "C" __global__ void silu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float sig = 1.0f / (1.0f + expf(-v));
    out[idx] = v * sig;
}

extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

// Phase 4 chunk 4.2 SiLU backward. Per-cell:
//   sig    = 1 / (1 + exp(-x))
//   dsilu  = sig * (1 + x * (1 - sig))
//   out[i] = grad_y[i] * dsilu(x[i])
// Matches the CPU `Tensor::silu_backward` cell-for-cell. One thread per
// element; same launch shape as forward `silu_f32`.
extern "C" __global__ void silu_backward_f32(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = x[idx];
    float sig = 1.0f / (1.0f + expf(-xv));
    float dsilu = sig * (1.0f + xv * (1.0f - sig));
    out[idx] = grad_y[idx] * dsilu;
}

// Phase 4 chunk 4.3 softmax backward. Per-row:
//   dot_i         = sum_k grad_y[i, k] * s[i, k]
//   grad_in[i, j] = s[i, j] * (grad_y[i, j] - dot_i)
// `s` is the saved softmax forward output (autograd.rs:618-621). One
// thread per row, matching forward `softmax_row_f32`'s launch shape.
// Two sequential passes (compute dot, then per-cell update) trade
// intra-row parallelism for kernel simplicity at the model's seq_len
// (<= 128 v0.13, <= 128 shakespeare-large).
extern "C" __global__ void softmax_backward_row_f32(
    const float* __restrict__ grad_y,
    const float* __restrict__ s,
    float* __restrict__ out,
    int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float* gr = grad_y + row * n;
    const float* sr = s + row * n;
    float* or_ = out + row * n;

    float dot = 0.0f;
    for (int k = 0; k < n; ++k) dot += gr[k] * sr[k];
    for (int j = 0; j < n; ++j) or_[j] = sr[j] * (gr[j] - dot);
}

// Phase 4 chunk 4.3 causal-mask backward. Lower triangle (j <= i)
// passes through unchanged; upper triangle is zeroed (the forward
// overwrote those cells with -inf, contributing no gradient). No saved
// tensor: the mask pattern is shape-determined. Same 2-D 16x16 launch
// shape as forward `causal_mask_f32`.
extern "C" __global__ void causal_mask_backward_f32(
    const float* __restrict__ grad_y,
    float* __restrict__ out,
    int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        out[i * n + j] = (j > i) ? 0.0f : grad_y[i * n + j];
    }
}

// Phase 4 chunk 4.4 RMSNorm backward. Per-row formula:
//   inv_rms_i  = 1 / sqrt(mean_j(x_saved[i, j]^2) + EPS)
//   dot_i      = sum_j x_saved[i, j] * grad_y[i, j]
//   factor_i   = dot_i * inv_rms_i^3 / n
//   grad_in[i, j] = grad_y[i, j] * inv_rms_i - x_saved[i, j] * factor_i
// EPS = 1e-5 matches the CPU helper exactly. One thread per row -
// same launch shape as forward `rmsnorm_row_f32`. Three sequential
// passes per row (mean_sq, dot, then per-cell update) keep the
// kernel simple at v0.13 hidden_dim 192 / shakespeare-large
// hidden_dim 256.
extern "C" __global__ void rmsnorm_backward_row_f32(
    const float* __restrict__ grad_y,
    const float* __restrict__ x_saved,
    float* __restrict__ out,
    int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float* gr = grad_y + row * n;
    const float* xr = x_saved + row * n;
    float* or_ = out + row * n;

    float sum_sq = 0.0f;
    float dot = 0.0f;
    for (int j = 0; j < n; ++j) {
        float xv = xr[j];
        sum_sq += xv * xv;
        dot += xv * gr[j];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + 1.0e-5f);
    float factor = dot * inv_rms * inv_rms * inv_rms / (float)n;
    for (int j = 0; j < n; ++j) {
        or_[j] = gr[j] * inv_rms - xr[j] * factor;
    }
}

// Phase 5.a absmean-ternary weight quant, STE forward - FUSED (issue
// #2). Per-tensor gamma = mean(|W|); output =
// clamp(round(W/(gamma+eps)), -1, +1) * gamma.
//
// The apply stage needs a reduction over the ENTIRE tensor, so the
// fusion runs as ONE cooperative block (grid must be exactly 1):
//   1. rows strided across threads, each row summed by a single
//      thread in j-order - the same per-row accumulation order as the
//      old two-stage kernel;
//   2. __syncthreads() replaces the old kernel boundary;
//   3. every thread re-sums row_abs_sum sequentially (r = 0..m, the
//      old apply kernel's order) so gamma is bit-identical, then the
//      cells are strided across threads.
// One block trades SM occupancy for a saved launch: at these tensor
// sizes the launch (~10-30 us) costs more than the whole kernel, and
// launch count - not compute - is the step's bottleneck. Matches
// `Tensor::quantise_weights_ste` byte-for-byte.
extern "C" __global__ void quantise_weights_ste_fused_f32(
    const float* __restrict__ w,
    float* __restrict__ row_abs_sum,
    float* __restrict__ out,
    int m, int n)
{
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        const float* wr = w + row * n;
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += fabsf(wr[j]);
        row_abs_sum[row] = s;
    }
    __syncthreads();
    float total_abs = 0.0f;
    for (int r = 0; r < m; ++r) total_abs += row_abs_sum[r];
    float gamma = total_abs / (float)(m * n);
    float denom = gamma + 1.0e-5f;
    int total = m * n;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        float v = w[idx] / denom;
        float q = roundf(v);
        if (q < -1.0f) q = -1.0f;
        else if (q > 1.0f) q = 1.0f;
        out[idx] = q * gamma;
    }
}

// Phase 5.a absmax-INT8 per-row activation quant, STE forward - FUSED
// (issue #2). Per row: alpha = max_j |x[i, j]|; x_q =
// clamp(round(x * 127/alpha), -128, +127); out = (alpha/127) * x_q.
// Edge case: alpha = 0 yields out = 0 directly (no NaN).
//
// alpha is per-row (no cross-row dependency), so the fusion runs one
// block PER ROW (grid = m blocks of QUANT_BLOCK threads): block-local
// strided absmax into shared memory, tree-reduce (max is exact -
// reassociation cannot change it), then the same block applies its
// row's cells. Matches `Tensor::quantise_acts_ste` byte-for-byte.
extern "C" __global__ void quantise_acts_ste_fused_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int m, int n)
{
    int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + row * n;
    float* or_ = out + row * n;
    __shared__ float smax[256];
    float a = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = fabsf(xr[j]);
        if (v > a) a = v;
    }
    smax[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x]) {
            smax[threadIdx.x] = smax[threadIdx.x + s];
        }
        __syncthreads();
    }
    float alpha = smax[0];
    if (alpha == 0.0f) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) or_[j] = 0.0f;
        return;
    }
    float scale_to_int = 127.0f / alpha;
    float row_dequant = alpha * (1.0f / 127.0f);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = xr[j] * scale_to_int;
        float q = roundf(v);
        if (q < -128.0f) q = -128.0f;
        else if (q > 127.0f) q = 127.0f;
        or_[j] = q * row_dequant;
    }
}

// Phase 5.b: int8 BitLinear forward. Two fused quant kernels (issue
// #2) feed the cublasGemmEx int8 GEMM path, plus the dequant step:
//
//   1. `quantise_weights_int8_fused` - same one-cooperative-block
//      fusion as the f32 STE kernel, but writes int8 ternary
//      {-1, 0, +1} plus a single-cell gamma scalar instead of a
//      dequantised f32 tensor.
//   2. `quantise_acts_int8_fused` - same block-per-row fusion as the
//      f32 STE kernel, but writes int8 in [-128, 127] and keeps
//      alpha[m] as a separate f32 buffer.
//   3. `scale_int32_to_f32` - the dequantisation step after the
//      int8 GEMM: y[i, j] = c_int[i, j] * (alpha[i] * gamma / 127).

// Fused ternary weight quant, int8 output. ONE cooperative block
// (grid must be 1); see `quantise_weights_ste_fused_f32` for the
// stage layout and the bit-identical-gamma argument. gamma lands in
// a single-element f32 buffer for the dequant kernel.
extern "C" __global__ void quantise_weights_int8_fused(
    const float* __restrict__ w,
    float* __restrict__ row_abs_sum,
    signed char* __restrict__ w_q_out,
    float* __restrict__ gamma_out,
    int m, int n)
{
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        const float* wr = w + row * n;
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += fabsf(wr[j]);
        row_abs_sum[row] = s;
    }
    __syncthreads();
    float total_abs = 0.0f;
    for (int r = 0; r < m; ++r) total_abs += row_abs_sum[r];
    float gamma = total_abs / (float)(m * n);
    float denom = gamma + 1.0e-5f;
    if (threadIdx.x == 0) gamma_out[0] = gamma;
    int total = m * n;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        float v = w[idx] / denom;
        float q = roundf(v);
        if (q < -1.0f) q = -1.0f;
        else if (q > 1.0f) q = 1.0f;
        w_q_out[idx] = (signed char)((int)q);
    }
}

// Fused per-row INT8 activation quant: one block per row (grid = m);
// see `quantise_acts_ste_fused_f32` for the reduction layout. Writes
// raw int8 (no dequant) + alpha[m] for the post-GEMM scale kernel.
extern "C" __global__ void quantise_acts_int8_fused(
    const float* __restrict__ x,
    signed char* __restrict__ x_q_out,
    float* __restrict__ alpha_out,
    int m, int n)
{
    int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + row * n;
    signed char* qr = x_q_out + row * n;
    __shared__ float smax[256];
    float a = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = fabsf(xr[j]);
        if (v > a) a = v;
    }
    smax[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x]) {
            smax[threadIdx.x] = smax[threadIdx.x + s];
        }
        __syncthreads();
    }
    float alpha = smax[0];
    if (threadIdx.x == 0) alpha_out[row] = alpha;
    if (alpha == 0.0f) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) qr[j] = 0;
        return;
    }
    float scale_to_int = 127.0f / alpha;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = xr[j] * scale_to_int;
        float q = roundf(v);
        if (q < -128.0f) q = -128.0f;
        else if (q > 127.0f) q = 127.0f;
        qr[j] = (signed char)((int)q);
    }
}

// Phase 5.b: dequantise the int32 cublasGemmEx output back to f32.
// y[i, j] = c_int[i, j] * (alpha[i] * gamma_scalar / 127).
// One thread per output cell; alpha is a [m] buffer, gamma is a
// 1-element buffer read once per thread.
extern "C" __global__ void scale_int32_to_f32(
    const int* __restrict__ c_int,
    const float* __restrict__ alpha,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    float row_scale = alpha[i] * gamma[0] * (1.0f / 127.0f);
    out[i * n + j] = (float)c_int[i * n + j] * row_scale;
}

// Phase 4 chunk 4.5.d softmax + cross-entropy fused forward. Per-row
// (one thread per row): subtract-max log-sum-exp softmax + per-row
// loss `-(logits[target] - log_denom)`. Writes softmax_out [seq, vocab]
// and per_row_loss [seq]. The CPU caller averages per_row_loss after
// D->H (one f32 sum is far cheaper than fighting GPU atomic-add).
extern "C" __global__ void cross_entropy_softmax_loss_f32(
    const float* __restrict__ logits,
    const int*   __restrict__ targets,
    float* __restrict__ softmax_out,
    float* __restrict__ per_row_loss,
    int seq, int vocab)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq) return;
    const float* lr = logits + row * vocab;
    float* sr = softmax_out + row * vocab;

    float row_max = -INFINITY;
    for (int j = 0; j < vocab; ++j) {
        float v = lr[j];
        if (v > row_max) row_max = v;
    }
    float denom = 0.0f;
    for (int j = 0; j < vocab; ++j) {
        float e = expf(lr[j] - row_max);
        sr[j] = e;
        denom += e;
    }
    float log_denom = row_max + logf(denom);
    float inv = 1.0f / denom;
    for (int j = 0; j < vocab; ++j) sr[j] *= inv;
    int t = targets[row];
    per_row_loss[row] = -(lr[t] - log_denom);
}

// Phase 4 chunk 4.5.d cross-entropy backward. Per-cell:
//   grad_logits[i, j] = (softmax_saved[i, j] - (j == targets[i] ? 1.0 : 0.0)) / seq
// 2-D 16x16 launch: one thread per (row, col). seq is read from an
// argument so the kernel does not need a separate pre-divided inv_seq.
extern "C" __global__ void cross_entropy_backward_f32(
    const float* __restrict__ softmax_saved,
    const int*   __restrict__ targets,
    float* __restrict__ grad_logits,
    int seq, int vocab)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= seq || j >= vocab) return;
    float s = softmax_saved[i * vocab + j];
    float v = (j == targets[i]) ? (s - 1.0f) : s;
    grad_logits[i * vocab + j] = v / (float)seq;
}

// Phase 4 chunk 4.4 RoPE backward. Inverse rotation per (pos, pair):
// same trig table as forward, sign of `sin` flipped because each
// per-pair rotation is orthogonal:
//   grad_in[pos, 2i]   =  grad_y[pos, 2i]   * cos + grad_y[pos, 2i+1] * sin
//   grad_in[pos, 2i+1] = -grad_y[pos, 2i]   * sin + grad_y[pos, 2i+1] * cos
// No saved tensor: angles are shape-determined. Same 2-D launch
// shape as forward `rope_f32` (one thread per (pos, pair) pair).
extern "C" __global__ void rope_backward_f32(
    const float* __restrict__ grad_y,
    float* __restrict__ out,
    int seq, int head_dim)
{
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = blockIdx.y * blockDim.y + threadIdx.y;
    int half = head_dim / 2;
    if (pair >= half || pos >= seq) return;
    float theta_i = powf(10000.0f, -(2.0f * (float)pair) / (float)head_dim);
    float angle = (float)pos * theta_i;
    float c = cosf(angle);
    float s = sinf(angle);
    float ga = grad_y[pos * head_dim + 2 * pair];
    float gb = grad_y[pos * head_dim + 2 * pair + 1];
    out[pos * head_dim + 2 * pair]     =  ga * c + gb * s;
    out[pos * head_dim + 2 * pair + 1] = -ga * s + gb * c;
}

extern "C" __global__ void rmsnorm_row_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int m, int n)
{
    // One thread per row. Two passes: sum-of-squares then normalise.
    // EPS = 1e-5 matches the CPU helper exactly so checkpoints trained
    // with one path are numerically valid through the other.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float* xr = x + row * n;
    float* yr = out + row * n;

    float sum_sq = 0.0f;
    for (int j = 0; j < n; ++j) {
        float v = xr[j];
        sum_sq += v * v;
    }
    float mean_sq = sum_sq / (float)n;
    float inv_rms = 1.0f / sqrtf(mean_sq + 1.0e-5f);
    for (int j = 0; j < n; ++j) {
        yr[j] = xr[j] * inv_rms;
    }
}
"#;

#[cfg(test)]
const MATMUL_KERNEL_SRC: &str = r#"
extern "C" __global__ void matmul_f32(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int m, int k, int n)
{
    constexpr int TILE = 16;
    __shared__ float ls[TILE][TILE];
    __shared__ float rs[TILE][TILE];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int tiles = (k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int kk_l = t * TILE + tx;
        const int kk_r = t * TILE + ty;
        ls[ty][tx] = (row < m && kk_l < k) ? lhs[row * k + kk_l] : 0.0f;
        rs[ty][tx] = (kk_r < k && col < n) ? rhs[kk_r * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += ls[ty][i] * rs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}
"#;

/// Per-process CUDA state: device 0's context + default stream + the
/// cuBLAS handle bound to that stream + handles for every NVRTC-
/// compiled production kernel (Phase 2.2 added six op kernels). In
/// test builds we also compile + cache the hand-rolled tile-GEMM
/// kernel so cuBLAS sgemm can be cross-checked against an independent
/// matmul implementation.
pub struct CudaContextHolder {
    /// Held for destructor lifetime: the `Arc<CudaStream>`, `CudaBlas`
    /// handle, and `CudaFunction` handles all reference this context
    /// but are not strong owners. If `ctx` were dropped first they
    /// would dangle. Not read directly outside this module.
    #[allow(dead_code)]
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,

    /// Phase 2.2 / 2.3 op kernels (compiled together from `KERNELS_SRC`):
    pub add_fn: CudaFunction,
    pub mul_scalar_fn: CudaFunction,
    pub transpose_2d_fn: CudaFunction,
    pub causal_mask_fn: CudaFunction,
    pub softmax_row_fn: CudaFunction,
    pub rope_fn: CudaFunction,
    pub silu_fn: CudaFunction,
    pub silu_backward_fn: CudaFunction,
    pub mul_fn: CudaFunction,
    pub rmsnorm_row_fn: CudaFunction,
    pub softmax_backward_row_fn: CudaFunction,
    pub causal_mask_backward_fn: CudaFunction,
    pub rmsnorm_backward_row_fn: CudaFunction,
    pub rope_backward_fn: CudaFunction,
    pub cross_entropy_softmax_loss_fn: CudaFunction,
    pub cross_entropy_backward_fn: CudaFunction,
    pub quantise_weights_ste_fused_fn: CudaFunction,
    pub quantise_acts_ste_fused_fn: CudaFunction,
    pub quantise_weights_int8_fused_fn: CudaFunction,
    pub quantise_acts_int8_fused_fn: CudaFunction,
    pub scale_int32_to_f32_fn: CudaFunction,

    /// Hand-rolled tile-based GEMM kernel from `MATMUL_KERNEL_SRC`.
    /// Test-only - the production matmul path uses cuBLAS sgemm (much
    /// better tuned across shapes than a single fixed 16x16 tile
    /// kernel). Useful for asserting cuBLAS correctness against an
    /// independent code path.
    #[cfg(test)]
    pub matmul_fn: CudaFunction,
}

thread_local! {
    /// Per-thread stream/cuBLAS override (issue #3, CUDA graphs).
    /// When set, every CUDA op issued from THIS thread enqueues onto
    /// the override stream (and routes GEMMs through the override
    /// cuBLAS handle) instead of the process-wide defaults. Graph
    /// capture uses this so a training step can be recorded on a
    /// private stream while other threads keep using the default
    /// stream completely unaffected - work from another thread can
    /// neither leak into the captured graph nor be blocked by the
    /// capture (mode `THREAD_LOCAL`).
    static STREAM_OVERRIDE: std::cell::RefCell<Option<(Arc<CudaStream>, Arc<CudaBlas>)>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard installing the thread-local stream/cuBLAS override;
/// restores the defaults on drop (including on unwind, so a panicked
/// capture cannot leave later ops on this thread pointing at the
/// capture stream).
struct StreamOverrideGuard;

impl StreamOverrideGuard {
    fn install(stream: Arc<CudaStream>, blas: Arc<CudaBlas>) -> Self {
        STREAM_OVERRIDE.with(|o| *o.borrow_mut() = Some((stream, blas)));
        StreamOverrideGuard
    }
}

impl Drop for StreamOverrideGuard {
    fn drop(&mut self) {
        STREAM_OVERRIDE.with(|o| *o.borrow_mut() = None);
    }
}

impl CudaContextHolder {
    /// The stream ops on this thread must enqueue onto: the
    /// thread-local capture override when installed, else the shared
    /// default stream. Every launch/alloc/memcpy in this module goes
    /// through here so a graph capture sees the whole op sequence.
    fn active_stream(&self) -> Arc<CudaStream> {
        STREAM_OVERRIDE
            .with(|o| o.borrow().as_ref().map(|(s, _)| s.clone()))
            .unwrap_or_else(|| self.stream.clone())
    }

    /// cuBLAS handle counterpart of `active_stream`: GEMMs must run on
    /// the same stream as the surrounding kernels or a captured graph
    /// would miss them (a cuBLAS handle is bound to one stream).
    fn with_active_blas<R>(&self, f: impl FnOnce(&CudaBlas) -> R) -> R {
        let over = STREAM_OVERRIDE.with(|o| o.borrow().as_ref().map(|(_, b)| b.clone()));
        match over {
            Some(b) => f(&b),
            None => f(&self.blas),
        }
    }
}

/// Lazy global handle. First call attaches to GPU 0, instantiates a
/// cuBLAS handle on the default stream, and NVRTC-compiles the
/// production kernel module (Phase 2.2 op kernels). In test builds it
/// also compiles the hand-rolled GEMM kernel into a separate module
/// for the cuBLAS-vs-NVRTC cross-check. Later calls return the cached
/// holder. The `Result` is itself cached, so a CUDA-less environment
/// fails fast on every call without re-trying the device-open each time.
pub fn cuda_state() -> Result<&'static CudaContextHolder, String> {
    static STATE: OnceLock<Result<CudaContextHolder, String>> = OnceLock::new();
    let r = STATE.get_or_init(|| {
        let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new(0) failed: {e:?}"))?;
        // Manual stream synchronisation (issue #3). cudarc's automatic
        // event tracking is dormant in single-stream mode, but the
        // moment `build_step_graph` creates its private capture stream
        // the tracking would activate and inject `cuStreamWaitEvent`
        // calls on events recorded before the capture started - which
        // invalidates CUDA-graph capture. Disabling it keeps every op
        // exactly as stream-ordered as it always was; the one real
        // cross-stream edge (default-stream weight uploads vs capture-
        // stream reads) is ordered explicitly by the graph code.
        // Safety (per cudarc docs): only affects slices created after
        // this call; nothing has been allocated yet.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let blas =
            CudaBlas::new(stream.clone()).map_err(|e| format!("CudaBlas::new failed: {e:?}"))?;

        // Phase 2.2 production kernel module: one NVRTC compile, six
        // function handles loaded out of the resulting module.
        let ops_ptx = compile_ptx(KERNELS_SRC)
            .map_err(|e| format!("NVRTC compile (KERNELS_SRC) failed: {e:?}"))?;
        let ops_module = ctx
            .load_module(ops_ptx)
            .map_err(|e| format!("load_module (ops) failed: {e:?}"))?;
        let load = |name: &str| -> Result<CudaFunction, String> {
            ops_module
                .load_function(name)
                .map_err(|e| format!("load_function({name:?}) failed: {e:?}"))
        };
        let add_fn = load("add_f32")?;
        let mul_scalar_fn = load("mul_scalar_f32")?;
        let transpose_2d_fn = load("transpose_2d_f32")?;
        let causal_mask_fn = load("causal_mask_f32")?;
        let softmax_row_fn = load("softmax_row_f32")?;
        let rope_fn = load("rope_f32")?;
        let silu_fn = load("silu_f32")?;
        let silu_backward_fn = load("silu_backward_f32")?;
        let mul_fn = load("mul_f32")?;
        let rmsnorm_row_fn = load("rmsnorm_row_f32")?;
        let softmax_backward_row_fn = load("softmax_backward_row_f32")?;
        let causal_mask_backward_fn = load("causal_mask_backward_f32")?;
        let rmsnorm_backward_row_fn = load("rmsnorm_backward_row_f32")?;
        let rope_backward_fn = load("rope_backward_f32")?;
        let cross_entropy_softmax_loss_fn = load("cross_entropy_softmax_loss_f32")?;
        let cross_entropy_backward_fn = load("cross_entropy_backward_f32")?;
        let quantise_weights_ste_fused_fn = load("quantise_weights_ste_fused_f32")?;
        let quantise_acts_ste_fused_fn = load("quantise_acts_ste_fused_f32")?;
        let quantise_weights_int8_fused_fn = load("quantise_weights_int8_fused")?;
        let quantise_acts_int8_fused_fn = load("quantise_acts_int8_fused")?;
        let scale_int32_to_f32_fn = load("scale_int32_to_f32")?;

        #[cfg(test)]
        let matmul_fn = {
            let ptx = compile_ptx(MATMUL_KERNEL_SRC)
                .map_err(|e| format!("NVRTC compile (MATMUL) failed: {e:?}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("load_module (matmul) failed: {e:?}"))?;
            module
                .load_function("matmul_f32")
                .map_err(|e| format!("load_function(\"matmul_f32\") failed: {e:?}"))?
        };
        Ok(CudaContextHolder {
            ctx,
            stream,
            blas,
            add_fn,
            mul_scalar_fn,
            transpose_2d_fn,
            causal_mask_fn,
            softmax_row_fn,
            rope_fn,
            silu_fn,
            silu_backward_fn,
            mul_fn,
            rmsnorm_row_fn,
            softmax_backward_row_fn,
            causal_mask_backward_fn,
            rmsnorm_backward_row_fn,
            rope_backward_fn,
            cross_entropy_softmax_loss_fn,
            cross_entropy_backward_fn,
            quantise_weights_ste_fused_fn,
            quantise_acts_ste_fused_fn,
            quantise_weights_int8_fused_fn,
            quantise_acts_int8_fused_fn,
            scale_int32_to_f32_fn,
            #[cfg(test)]
            matmul_fn,
        })
    });
    r.as_ref().map_err(|s| s.clone())
}

thread_local! {
    /// Count of quantisation kernel launches issued BY THIS THREAD
    /// (f32 STE traits + the int8 `bit_linear` quant stages). Issue #2
    /// observability: with thousands of launches per training step,
    /// launch overhead - not GEMM compute - is the GPU bottleneck, so
    /// tests pin the exact number of launches each quantise call may
    /// issue. Thread-local so parallel tests cannot perturb each
    /// other's measurements.
    pub static QUANT_KERNEL_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Increment helper for `QUANT_KERNEL_LAUNCHES` - one call per quant
/// kernel launch, next to the `launch` itself.
fn count_quant_launch() {
    QUANT_KERNEL_LAUNCHES.with(|c| c.set(c.get() + 1));
}

thread_local! {
    /// Count of device-to-host reads issued BY THIS THREAD through the
    /// step/readback paths (`to_cpu` + the step-loss/grad reads).
    /// Issue #15 observability: each read is a driver round-trip plus
    /// a pageable-copy sync, so tests pin how many a training step may
    /// issue. Thread-local for parallel-suite immunity, like
    /// `QUANT_KERNEL_LAUNCHES`.
    pub static DTOH_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Increment helper for `DTOH_READS` - one call per D->H copy, next to
/// the copy itself.
fn count_dtoh_read() {
    DTOH_READS.with(|c| c.set(c.get() + 1));
}

/// Device-resident f32 tensor. Owns its `CudaSlice<f32>`; the slice is
/// reference-counted by cudarc so cloning is cheap on the Rust side and
/// safe across stream boundaries.
pub struct CudaTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
}

/// Deep-copy a `CudaTensor` (alloc + device-to-device memcpy) so two
/// independent backward branches can flow incoming gradients through
/// without aliasing the same device buffer. Used by Phase 4 backward
/// helpers (e.g. `add_backward` returns two clones of grad_c).
/// Errors from cudarc are panics here, matching the rest of the file's
/// invariant-style error handling.
impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        let s = cuda_state().expect("cuda_state failed");
        let new_data = s
            .active_stream()
            .clone_dtod(&self.data)
            .expect("CudaTensor clone (clone_dtod) failed");
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }
}

impl CudaTensor {
    /// Copy a CPU `Tensor` into device memory. Synchronous on the default
    /// stream; the returned `CudaTensor` is ready for kernel launches.
    pub fn from_cpu(t: &Tensor) -> Result<Self, String> {
        let s = cuda_state()?;
        let data = s
            .active_stream()
            .clone_htod(&t.data)
            .map_err(|e| format!("clone_htod failed: {e:?}"))?;
        Ok(Self {
            data,
            shape: t.shape.clone(),
        })
    }

    /// Overwrite this tensor's device buffer in place from a CPU tensor
    /// of identical shape. No allocation: the existing `CudaSlice` is
    /// reused via `memcpy_htod` (stream-ordered, so later kernel
    /// launches on the same stream see the new values).
    pub fn copy_from_cpu(&mut self, t: &Tensor) -> Result<(), String> {
        if self.shape != t.shape {
            return Err(format!(
                "copy_from_cpu shape mismatch: device {:?} vs host {:?}",
                self.shape, t.shape
            ));
        }
        let s = cuda_state()?;
        s.active_stream()
            .memcpy_htod(&t.data, &mut self.data)
            .map_err(|e| format!("memcpy_htod failed: {e:?}"))
    }

    /// Copy device memory back into a CPU `Tensor`. Synchronous.
    pub fn to_cpu(&self) -> Result<Tensor, String> {
        let s = cuda_state()?;
        count_dtoh_read();
        let v = s
            .active_stream()
            .clone_dtoh(&self.data)
            .map_err(|e| format!("clone_dtoh failed: {e:?}"))?;
        Ok(Tensor::from_vec(v, self.shape.clone()))
    }

    /// Hand-rolled tile-based GEMM via the v0.18 NVRTC kernel. Kept as a
    /// reference implementation that we can A/B against the cuBLAS path
    /// in tests and benchmarks. Production callers should use the
    /// `MatMul::matmul` trait method (which dispatches to cuBLAS); this
    /// method is `#[cfg(test)]` to avoid surface-area drift while still
    /// letting the test suite exercise the bespoke kernel.
    #[cfg(test)]
    pub fn matmul_nvrtc(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(rhs.shape.len(), 2);
        assert_eq!(self.shape[1], rhs.shape[0]);
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let k_i = i32::try_from(k).expect("k exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out: CudaSlice<f32> = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(TILE), (m_i as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut launcher = stream.launch_builder(&s.matmul_fn);
        launcher.arg(&self.data);
        launcher.arg(&rhs.data);
        launcher.arg(&mut out);
        launcher.arg(&m_i);
        launcher.arg(&k_i);
        launcher.arg(&n_i);
        // Safety: same as the v0.18 kernel - signature matches the six
        // args pushed above; output buffer sized m*n; lhs / rhs sized
        // m*k and k*n.
        unsafe { launcher.launch(cfg) }.expect("NVRTC kernel launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

/// Production GEMM via cuBLAS sgemm. Output shape `[m, n]` from inputs
/// `[m, k]` and `[k, n]`. Synchronises the stream before returning so
/// `to_cpu()` immediately afterwards is safe; in Phase 2+ we will
/// compose multiple cuBLAS / kernel calls without intermediate sync.
///
/// **Row-major-via-column-major trick.** cuBLAS uses Fortran (column-
/// major) storage: a (rows, cols) col-major matrix has element
/// `[r, c]` at index `c * rows + r`. Our tensors are row-major. The
/// trick: a row-major matrix viewed as col-major has its dimensions
/// transposed (the same bytes describe `(M, N)` row-major *and* `(N,
/// M)` col-major). So to compute `C_row = A_row @ B_row` of shape
/// `(M, N)` we ask cuBLAS for `C_col = B_col @ A_col` of shape `(N,
/// M)` - the bytes that come out match the C_row layout exactly.
/// Passing B before A and swapping (m, n) is the entire adapter; no
/// transpose flags are needed.
///
/// Errors from cuBLAS / cudarc (bad shape, lost device, allocation
/// failure) are panics here; callers have no recovery path and the
/// rest of the project is panic-on-invariant style.
impl crate::device::MatMul for CudaTensor {
    fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape.len(), 2, "lhs must be 2-D, got {:?}", self.shape);
        assert_eq!(rhs.shape.len(), 2, "rhs must be 2-D, got {:?}", rhs.shape);
        assert_eq!(
            self.shape[1], rhs.shape[0],
            "shape mismatch: {:?} @ {:?}",
            self.shape, rhs.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let k_i = i32::try_from(k).expect("k exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");

        let mut out: CudaSlice<f32> = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");

        // Compute C_col = B_col @ A_col of shape (N, M).
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            // cuBLAS-side dimensions: m_cublas=N, n_cublas=M, k_cublas=K.
            m: n_i,
            n: m_i,
            k: k_i,
            alpha: 1.0,
            // Leading dim of B in col-major is N (matches row-major's
            // stride - both views read N f32 values per "row" of B).
            lda: n_i,
            // Leading dim of A in col-major is K.
            ldb: k_i,
            beta: 0.0,
            // Leading dim of C in col-major is N (= row-major's stride).
            ldc: n_i,
        };

        // Safety: shapes / strides above match the row-major <-> col-major
        // adapter described in the doc comment; alloc_zeros above gave us
        // exactly m*n f32 of writable device memory; self.data and
        // rhs.data are owned device slices of size m*k and k*n.
        s.with_active_blas(|blas| unsafe { blas.gemm(cfg, &rhs.data, &self.data, &mut out) })
            .expect("cuBLAS sgemm failed");

        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

// ---- Phase 2.2 op trait impls for CudaTensor.
//
// Pattern: derive launch dimensions from `self.shape`, alloc the
// output, push args via `launch_builder`, launch unsafe, synchronise
// the stream (so callers can `to_cpu()` immediately or pass to the
// next op without explicit barrier). cudarc errors `.expect()` at the
// boundary - same panic semantics as the matmul impl above.
//
// Block dim choices:
//   add / mul_scalar           : 1-D grid, 256 threads/block
//   transpose_2d / causal_mask : 2-D grid, 16x16 threads/block
//   softmax_row                : 1-D grid over rows, 256 threads/block
//   rope                       : 2-D grid over (pair, pos), 16x16 block

impl crate::device::Add for CudaTensor {
    fn add(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape, rhs.shape,
            "add: shape mismatch: {:?} vs {:?}",
            self.shape, rhs.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.add_fn);
        l.arg(&self.data);
        l.arg(&rhs.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: kernel signature (const float*, const float*, float*,
        // int) matches the four args; output is sized n; lhs / rhs are
        // both sized n (asserted above).
        unsafe { l.launch(cfg) }.expect("add_f32 launch failed");
        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }
}

impl crate::device::MulScalar for CudaTensor {
    fn mul_scalar(&self, s: f32) -> Self {
        let st = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = st
            .active_stream()
            .alloc_zeros::<f32>(n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = st.active_stream();
        let mut l = stream.launch_builder(&st.mul_scalar_fn);
        l.arg(&self.data);
        l.arg(&s);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float*, float, float*, int) matches
        // the four args; output sized n; input sized n.
        unsafe { l.launch(cfg) }.expect("mul_scalar_f32 launch failed");
        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }
}

impl crate::device::Transpose2D for CudaTensor {
    fn transpose_2d(&self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "transpose_2d: rank-2 only, got {:?}",
            self.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let r = self.shape[0];
        let c = self.shape[1];
        let r_i = i32::try_from(r).expect("r exceeds i32");
        let c_i = i32::try_from(c).expect("c exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(r * c)
            .expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: ((r_i as u32).div_ceil(TILE), (c_i as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.transpose_2d_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&r_i);
        l.arg(&c_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; input sized r*c; output sized r*c (= c*r).
        unsafe { l.launch(cfg) }.expect("transpose_2d_f32 launch failed");
        Self {
            data: out,
            shape: vec![c, r],
        }
    }
}

impl crate::device::CausalMask for CudaTensor {
    fn causal_mask(&self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "causal_mask: rank-2 only, got {:?}",
            self.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(TILE), (m_i as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.causal_mask_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized m*n.
        unsafe { l.launch(cfg) }.expect("causal_mask_f32 launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::Softmax for CudaTensor {
    fn softmax(&self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "softmax: rank-2 only, got {:?}",
            self.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        // One thread per row. 256 rows per block; covers shakespeare
        // configs without needing a multi-thread-per-row reduction.
        let cfg = LaunchConfig {
            grid_dim: ((m_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.softmax_row_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized m*n.
        unsafe { l.launch(cfg) }.expect("softmax_row_f32 launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::SoftmaxBackward for CudaTensor {
    fn softmax_backward(&self, s_out: &Self) -> Self {
        assert_eq!(
            self.shape, s_out.shape,
            "softmax_backward: grad_y / s_out shape mismatch ({:?} vs {:?})",
            self.shape, s_out.shape,
        );
        assert_eq!(self.shape.len(), 2, "softmax_backward: rank-2 only");
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((m_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.softmax_backward_row_fn);
        l.arg(&self.data);
        l.arg(&s_out.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float* grad_y, const float* s,
        // float* out, int m, int n) matches the five args; all three
        // buffers sized m*n; output freshly allocated above.
        unsafe { l.launch(cfg) }.expect("softmax_backward_row_f32 launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::CausalMaskBackward for CudaTensor {
    fn causal_mask_backward(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "causal_mask_backward: rank-2 only");
        let st = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = st
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        // 16x16 threads per block matches the forward `causal_mask_f32`
        // launch shape. One thread per (i, j) output cell.
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(16), (m_i as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let stream = st.active_stream();
        let mut l = stream.launch_builder(&st.causal_mask_backward_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float* grad_y, float* out, int m,
        // int n) matches the four args; both buffers sized m*n.
        unsafe { l.launch(cfg) }.expect("causal_mask_backward_f32 launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::Silu for CudaTensor {
    fn silu(&self) -> Self {
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.silu_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int) matches the
        // three args; both buffers sized n.
        unsafe { l.launch(cfg) }.expect("silu_f32 launch failed");
        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }
}

impl crate::device::SiluBackward for CudaTensor {
    fn silu_backward(&self, x: &Self) -> Self {
        assert_eq!(
            self.shape, x.shape,
            "silu_backward: grad_y and x shape mismatch ({:?} vs {:?})",
            self.shape, x.shape,
        );
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.silu_backward_fn);
        l.arg(&self.data);
        l.arg(&x.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float* grad_y, const float* x,
        // float* out, int n) matches the four args; all three buffers
        // sized n; output freshly allocated above.
        unsafe { l.launch(cfg) }.expect("silu_backward_f32 launch failed");
        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }
}

impl crate::device::RmsNorm for CudaTensor {
    fn rmsnorm(&self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "rmsnorm: rank-2 only, got {:?}",
            self.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((m_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.rmsnorm_row_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized m*n.
        unsafe { l.launch(cfg) }.expect("rmsnorm_row_f32 launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::RmsNormBackward for CudaTensor {
    fn rmsnorm_backward(&self, x_saved: &Self) -> Self {
        assert_eq!(
            self.shape, x_saved.shape,
            "rmsnorm_backward: grad_y / x_saved shape mismatch ({:?} vs {:?})",
            self.shape, x_saved.shape,
        );
        assert_eq!(self.shape.len(), 2, "rmsnorm_backward: rank-2 only");
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        // Same launch shape as forward `rmsnorm_row_f32`: one thread per row.
        let cfg = LaunchConfig {
            grid_dim: ((m_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.rmsnorm_backward_row_fn);
        l.arg(&self.data);
        l.arg(&x_saved.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float* grad_y, const float* x_saved,
        // float* out, int m, int n) matches the five args; all three
        // buffers sized m*n; output freshly allocated above.
        unsafe { l.launch(cfg) }.expect("rmsnorm_backward_row_f32 launch failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::Mul for CudaTensor {
    fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape, rhs.shape,
            "mul: shape mismatch: {:?} vs {:?}",
            self.shape, rhs.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(n)
            .expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.mul_fn);
        l.arg(&self.data);
        l.arg(&rhs.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float*, const float*, float*, int)
        // matches the four args; output sized n; lhs / rhs both sized n
        // (asserted above).
        unsafe { l.launch(cfg) }.expect("mul_f32 launch failed");
        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }
}

impl crate::device::Rope for CudaTensor {
    fn rope(&self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "rope: rank-2 only, got {:?}",
            self.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let seq = self.shape[0];
        let head_dim = self.shape[1];
        assert!(
            head_dim.is_multiple_of(2),
            "rope: head_dim ({head_dim}) must be even"
        );
        let seq_i = i32::try_from(seq).expect("seq exceeds i32");
        let hd_i = i32::try_from(head_dim).expect("head_dim exceeds i32");
        let half_i = hd_i / 2;
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(seq * head_dim)
            .expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: (
                (half_i as u32).div_ceil(TILE),
                (seq_i as u32).div_ceil(TILE),
                1,
            ),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.rope_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&seq_i);
        l.arg(&hd_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized seq*head_dim.
        unsafe { l.launch(cfg) }.expect("rope_f32 launch failed");
        Self {
            data: out,
            shape: vec![seq, head_dim],
        }
    }
}

impl crate::device::RopeBackward for CudaTensor {
    fn rope_backward(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "rope_backward: rank-2 only");
        let s = cuda_state().expect("cuda_state failed");
        let seq = self.shape[0];
        let head_dim = self.shape[1];
        assert!(
            head_dim.is_multiple_of(2),
            "rope_backward: head_dim ({head_dim}) must be even"
        );
        let seq_i = i32::try_from(seq).expect("seq exceeds i32");
        let hd_i = i32::try_from(head_dim).expect("head_dim exceeds i32");
        let half_i = hd_i / 2;
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(seq * head_dim)
            .expect("alloc_zeros failed");
        // Same 2-D 16x16 launch shape as forward `rope_f32` - one
        // thread per (pos, pair).
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: (
                (half_i as u32).div_ceil(TILE),
                (seq_i as u32).div_ceil(TILE),
                1,
            ),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.rope_backward_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&seq_i);
        l.arg(&hd_i);
        // Safety: signature (const float* grad_y, float* out, int seq,
        // int head_dim) matches the four args; both buffers sized
        // seq*head_dim; output freshly allocated above.
        unsafe { l.launch(cfg) }.expect("rope_backward_f32 launch failed");
        Self {
            data: out,
            shape: vec![seq, head_dim],
        }
    }
}

impl CudaTensor {
    /// Device-resident cross-entropy forward: fused softmax + per-row
    /// loss, targets already on the device, NO host transfer. Returns
    /// the `[seq]` per-row loss buffer and the saved softmax. This is
    /// the capture-safe core (issue #3): the trait method wraps it
    /// with the targets htod and the host-side mean.
    fn cross_entropy_forward_device(
        &self,
        targets_dev: &CudaSlice<i32>,
    ) -> (CudaSlice<f32>, CudaTensor) {
        assert_eq!(self.shape.len(), 2, "cross_entropy: rank-2 logits");
        let seq = self.shape[0];
        let vocab = self.shape[1];
        assert_eq!(targets_dev.len(), seq, "cross_entropy: target len mismatch");
        let s = cuda_state().expect("cuda_state failed");
        let mut softmax = s
            .active_stream()
            .alloc_zeros::<f32>(seq * vocab)
            .expect("alloc softmax");
        let mut per_row_loss = s
            .active_stream()
            .alloc_zeros::<f32>(seq)
            .expect("alloc loss");
        let seq_i = i32::try_from(seq).expect("seq exceeds i32");
        let vocab_i = i32::try_from(vocab).expect("vocab exceeds i32");
        // One thread per row, 256 rows per block.
        let cfg = LaunchConfig {
            grid_dim: ((seq_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.cross_entropy_softmax_loss_fn);
        l.arg(&self.data);
        l.arg(targets_dev);
        l.arg(&mut softmax);
        l.arg(&mut per_row_loss);
        l.arg(&seq_i);
        l.arg(&vocab_i);
        // Safety: signature (const float* logits, const int* targets,
        // float* softmax, float* per_row_loss, int seq, int vocab)
        // matches; logits / softmax sized seq*vocab; loss sized seq.
        unsafe { l.launch(cfg) }.expect("cross_entropy_softmax_loss_f32 launch");
        (
            per_row_loss,
            CudaTensor {
                data: softmax,
                shape: vec![seq, vocab],
            },
        )
    }

    /// Host-target convenience used by both trait impls below: usize
    /// targets validated + converted + copied H->D once.
    fn targets_to_device(targets: &[usize], vocab: usize) -> CudaSlice<i32> {
        let s = cuda_state().expect("cuda_state failed");
        // Targets to device as i32 (kernel reads `int*`); usize -> i32
        // panics on out-of-range (only realistic at vocab >= 2^31, which
        // the rest of the code wouldn't survive either).
        let targets_i32: Vec<i32> = targets
            .iter()
            .map(|&t| {
                assert!(t < vocab, "cross_entropy: target {t} >= vocab {vocab}");
                i32::try_from(t).expect("target exceeds i32")
            })
            .collect();
        s.active_stream()
            .clone_htod(&targets_i32)
            .expect("clone_htod targets")
    }
}

impl crate::device::CrossEntropy for CudaTensor {
    fn cross_entropy_forward_save(&self, targets: &[usize]) -> (f32, Self) {
        assert_eq!(self.shape.len(), 2, "cross_entropy: rank-2 logits");
        let seq = self.shape[0];
        let vocab = self.shape[1];
        assert_eq!(targets.len(), seq, "cross_entropy: target len mismatch");
        let s = cuda_state().expect("cuda_state failed");
        let targets_dev = Self::targets_to_device(targets, vocab);
        let (per_row_loss, softmax) = self.cross_entropy_forward_device(&targets_dev);
        // Copy per-row losses back and average. seq is small (~64) so
        // this is far cheaper than fighting GPU atomic-add or a second
        // reduction kernel.
        count_dtoh_read();
        let losses_host = s
            .active_stream()
            .clone_dtoh(&per_row_loss)
            .expect("D->H per_row_loss");
        let total: f32 = losses_host.iter().sum();
        let loss = total / seq as f32;
        (loss, softmax)
    }
}

impl CudaTensor {
    /// Device-resident cross-entropy backward: targets already on the
    /// device, no host transfer. Capture-safe core (issue #3); the
    /// trait method wraps it with the targets htod.
    fn cross_entropy_backward_device(&self, targets_dev: &CudaSlice<i32>, seq: usize) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "cross_entropy_backward: rank-2 softmax"
        );
        assert_eq!(seq, self.shape[0], "cross_entropy_backward: seq mismatch");
        let vocab = self.shape[1];
        assert_eq!(
            targets_dev.len(),
            seq,
            "cross_entropy_backward: target len mismatch"
        );
        let s = cuda_state().expect("cuda_state failed");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(seq * vocab)
            .expect("alloc grad_logits");
        let seq_i = i32::try_from(seq).expect("seq exceeds i32");
        let vocab_i = i32::try_from(vocab).expect("vocab exceeds i32");
        // 2-D 16x16 launch: one thread per (row, col).
        let cfg = LaunchConfig {
            grid_dim: (
                (vocab_i as u32).div_ceil(16),
                (seq_i as u32).div_ceil(16),
                1,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.cross_entropy_backward_fn);
        l.arg(&self.data);
        l.arg(targets_dev);
        l.arg(&mut out);
        l.arg(&seq_i);
        l.arg(&vocab_i);
        // Safety: signature (const float* softmax_saved, const int*
        // targets, float* grad_logits, int seq, int vocab) matches;
        // both float buffers sized seq*vocab.
        unsafe { l.launch(cfg) }.expect("cross_entropy_backward_f32 launch");
        Self {
            data: out,
            shape: vec![seq, vocab],
        }
    }
}

impl crate::device::CrossEntropyBackward for CudaTensor {
    fn cross_entropy_backward(&self, targets: &[usize], seq: usize) -> Self {
        let vocab = self.shape[1];
        let targets_dev = Self::targets_to_device(targets, vocab);
        self.cross_entropy_backward_device(&targets_dev, seq)
    }
}

impl crate::device::QuantiseWeightsSTE for CudaTensor {
    fn quantise_weights_ste(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "quantise_weights_ste: rank-2 only");
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut row_abs_sum = s
            .active_stream()
            .alloc_zeros::<f32>(m)
            .expect("alloc row_abs_sum");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc out");
        // ONE fused launch (issue #2). The kernel is a single
        // cooperative block, so the grid MUST stay (1, 1, 1): a second
        // block would race the row_abs_sum scratch across the
        // __syncthreads() stage boundary.
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.quantise_weights_ste_fused_fn);
        l.arg(&self.data);
        l.arg(&mut row_abs_sum);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature matches; both buffers correctly sized.
        count_quant_launch();
        unsafe { l.launch(cfg) }.expect("quantise_weights_ste_fused launch");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::QuantiseActsSTE for CudaTensor {
    fn quantise_acts_ste(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "quantise_acts_ste: rank-2 only");
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc out");
        // ONE fused launch (issue #2): one block per row. The block
        // size must stay 256 to match the kernel's static smax[256].
        // No scratch buffer at all - the row reduction lives in
        // shared memory, so this also drops an alloc per call.
        let cfg = LaunchConfig {
            grid_dim: (m_i as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = s.active_stream();
        let mut l = stream.launch_builder(&s.quantise_acts_ste_fused_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        count_quant_launch();
        unsafe { l.launch(cfg) }.expect("quantise_acts_ste_fused launch");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

impl crate::device::BitLinear for CudaTensor {
    /// Phase 5.b: int8 BitLinear forward via cuBLAS `cublasGemmEx`
    /// (int8 inputs, int32 accumulator, tensor cores when shapes
    /// align). Algorithm:
    ///
    ///     y[m, n] = (alpha[m] * gamma / 127) * sum_k x_q[m, k] * w_q[k, n]
    ///
    /// where `x_q` is INT8 in [-128, 127] (per-row scale `alpha[m]`),
    /// `w_q` is INT8 in {-1, 0, +1} (scalar scale `gamma`). The
    /// expensive matmul runs in INT32 on tensor cores; the
    /// dequantisation kernel folds the scales back at the end. Same
    /// row-major-via-column-major adapter as the f32 `MatMul` impl
    /// (pass B and A swapped, swap m and n - cuBLAS sees row-major
    /// as transposed col-major).
    ///
    /// Mathematically equivalent to the Phase 5.a f32 path
    /// (`quantise_acts_ste(x).matmul(quantise_weights_ste(w))`); the
    /// only difference is round-off (the int32 accumulator is exact
    /// for integer multiplies, whereas the f32 matmul accumulates
    /// rounding error per term). On Ada tensor cores the int8 GEMM
    /// is ~10-50x faster than the f32 sgemm path at large matmul
    /// shapes; at small shapes / non-aligned dimensions cuBLAS may
    /// fall back to non-tensor-core paths and the speedup shrinks.
    ///
    /// **Shape fallback**: cuBLAS `cublasGemmEx` with int8 inputs
    /// requires `lda` and `ldb` to be multiples of 4 (the int8
    /// kernel reads 32-bit chunks). In our row-major-via-col-major
    /// adapter that means `n` and `k` must both be multiples of 4.
    /// In the project the only matmul that violates this is the
    /// lm_head with `n = vocab = 65`. When the shape check fails
    /// we fall back to the Phase 5.a f32 path
    /// (`quantise_acts_ste(x).matmul(quantise_weights_ste(w))`)
    /// instead of failing the GEMM call. The f32 path is
    /// algebraically identical so output values match.
    fn bit_linear(&self, rhs: &Self) -> Self {
        use cudarc::cublas::{result::gemm_ex, sys};
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use std::ffi::c_void;

        assert_eq!(
            self.shape.len(),
            2,
            "bit_linear lhs must be 2-D, got {:?}",
            self.shape
        );
        assert_eq!(
            rhs.shape.len(),
            2,
            "bit_linear rhs must be 2-D, got {:?}",
            rhs.shape
        );
        assert_eq!(
            self.shape[1], rhs.shape[0],
            "bit_linear shape mismatch: {:?} @ {:?}",
            self.shape, rhs.shape
        );
        let st = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let k_i = i32::try_from(k).expect("k exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");

        // Shape fallback: cuBLAS int8 GEMM requires lda and ldb to
        // be multiples of 4. In our row-major-via-col-major adapter
        // lda = n (B's row-major stride) and ldb = k (A's row-major
        // stride). When either fails the alignment check, fall back
        // to the Phase 5.a f32 sgemm path (algebraically identical;
        // the only matmul in the project that hits this fallback is
        // the lm_head with n = vocab = 65).
        if !k.is_multiple_of(4) || !n.is_multiple_of(4) {
            use crate::device::{MatMul, QuantiseActsSTE, QuantiseWeightsSTE};
            let x_eff = self.quantise_acts_ste();
            let w_eff = rhs.quantise_weights_ste();
            return x_eff.matmul(&w_eff);
        }

        // ---- Stage 1: quantise self (acts) to INT8 + alpha[m]. ----
        // ONE fused launch (issue #2): one block per row, block size
        // pinned to the kernel's static smax[256].
        let mut alpha = st
            .active_stream()
            .alloc_zeros::<f32>(m)
            .expect("alloc alpha");
        let mut x_q = st
            .active_stream()
            .alloc_zeros::<i8>(m * k)
            .expect("alloc x_q");
        let cfg_x = LaunchConfig {
            grid_dim: (m_i as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = st.active_stream();
        let mut l = stream.launch_builder(&st.quantise_acts_int8_fused_fn);
        l.arg(&self.data);
        l.arg(&mut x_q);
        l.arg(&mut alpha);
        l.arg(&m_i);
        l.arg(&k_i);
        count_quant_launch();
        unsafe { l.launch(cfg_x) }.expect("acts int8 fused launch");

        // ---- Stage 2: quantise rhs (weights) to INT8 + gamma scalar. ----
        // ONE fused launch (issue #2): a single cooperative block, so
        // the grid MUST stay (1, 1, 1) - see the kernel comment.
        let mut gamma = st
            .active_stream()
            .alloc_zeros::<f32>(1)
            .expect("alloc gamma");
        let mut w_row_abs_sum = st
            .active_stream()
            .alloc_zeros::<f32>(k)
            .expect("alloc w_row_abs_sum");
        let mut w_q = st
            .active_stream()
            .alloc_zeros::<i8>(k * n)
            .expect("alloc w_q");
        let cfg_w = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = st.active_stream();
        let mut l = stream.launch_builder(&st.quantise_weights_int8_fused_fn);
        l.arg(&rhs.data);
        l.arg(&mut w_row_abs_sum);
        l.arg(&mut w_q);
        l.arg(&mut gamma);
        l.arg(&k_i);
        l.arg(&n_i);
        count_quant_launch();
        unsafe { l.launch(cfg_w) }.expect("weights int8 fused launch");

        // ---- Stage 3: cublasGemmEx int8 -> int32. ----
        // Same row-major-via-column-major adapter as the f32 MatMul
        // impl: ask for C_col = B_col @ A_col of shape (N, M); the
        // bytes that come out match the desired C_row = A_row @ B_row.
        let mut c_int = st
            .active_stream()
            .alloc_zeros::<i32>(m * n)
            .expect("alloc c_int");
        let alpha_h: i32 = 1;
        let beta_h: i32 = 0;
        let blas_handle = st.with_active_blas(|blas| *blas.handle());
        // Safety: pointer arithmetic + raw FFI call. Buffers are
        // sized correctly above. Stride / shape / type tags below
        // match the col-major view of the row-major data.
        unsafe {
            let gemm_stream = st.active_stream();
            let (b_ptr, _b_keep) = w_q.device_ptr(&gemm_stream);
            let (a_ptr, _a_keep) = x_q.device_ptr(&gemm_stream);
            let (c_ptr, _c_keep) = c_int.device_ptr_mut(&gemm_stream);
            gemm_ex(
                blas_handle,
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n_i,
                m_i,
                k_i,
                &alpha_h as *const i32 as *const c_void,
                b_ptr as *const c_void,
                sys::cudaDataType::CUDA_R_8I,
                n_i,
                a_ptr as *const c_void,
                sys::cudaDataType::CUDA_R_8I,
                k_i,
                &beta_h as *const i32 as *const c_void,
                c_ptr as *mut c_void,
                sys::cudaDataType::CUDA_R_32I,
                n_i,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .expect("cublasGemmEx int8 failed");
        }

        // ---- Stage 4: dequantise int32 -> f32 with per-row alpha + scalar gamma. ----
        let mut out = st
            .active_stream()
            .alloc_zeros::<f32>(m * n)
            .expect("alloc out");
        let cfg_scale = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(16), (m_i as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let stream = st.active_stream();
        let mut l = stream.launch_builder(&st.scale_int32_to_f32_fn);
        l.arg(&c_int);
        l.arg(&alpha);
        l.arg(&gamma);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        unsafe { l.launch(cfg_scale) }.expect("scale_int32_to_f32 launch");

        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

// ---- Phase 3: end-to-end forward pass on the GPU.
//
// `CudaModel` mirrors the CPU `model::Model` but holds device-resident
// per-block weight bundles built on the Phase 2 generic structs
// (`HeadWeights<CudaTensor>` and `FfnWeights<CudaTensor>`). The token
// embedding table stays on the CPU side - one row-pick per input
// position is cheap enough that adding a dedicated embed-lookup kernel
// is not worth it; we just gather the [seq, hidden] slab on the host
// and copy it H->D once per forward call.
//
// Phase 3 is **inference-only** and **f32 throughout** - no BitNet
// ternary quant, no autograd. Comparing a CudaModel forward against
// the same architecture run through `block_inference<Tensor>` proves
// the device-side stack is correct end-to-end. Phase 4 will add per-op
// backward kernels and wire training; Phase 5 will add ternary
// tensor-core kernels for the quantised inference path.

/// Per-block GPU-resident weights. Built by `CudaModel::from_cpu` from
/// the corresponding `model::BlockMasters` on the host side.
pub struct CudaBlockMasters {
    pub heads: Vec<crate::device::HeadWeights<CudaTensor>>,
    pub ffn: crate::device::FfnWeights<CudaTensor>,
}

/// GPU-resident model. Token embedding is kept on the CPU side; every
/// other weight tensor lives on the device. End-to-end forward pass
/// runs entirely through the Phase 2 trait surface (matmul / rmsnorm /
/// silu / softmax / rope / etc.) so the same generic helpers are
/// exercised in production as in the per-op tests.
pub struct CudaModel {
    pub config: crate::model::ModelConfig,
    /// Kept on CPU: the embed step is a row-pick from a small table
    /// (vocab x hidden = 50 KB at v0.13 scale), so the simplest
    /// implementation is to gather the [seq, hidden] slab on host and
    /// copy it H->D once per forward call. Avoids an embed-lookup
    /// kernel at the cost of a tiny H->D transfer per call.
    pub token_embed_cpu: Tensor,
    /// Device-side copy of `token_embed`. Needed since v0.17 because the
    /// LM-head matmul reads `token_embed` (transposed) directly - tied
    /// embeddings. Synced from `token_embed_cpu` once per `from_cpu`
    /// call (which itself runs once per training step in the GPU
    /// path, so the device copy stays consistent with the master).
    pub token_embed_device: CudaTensor,
    pub blocks: Vec<CudaBlockMasters>,
}

impl CudaModel {
    /// Copy every parameter tensor from a CPU `Model` into device
    /// memory. The embed table is cloned to CPU (for the gather) AND
    /// to device (for the tied LM-head matmul). Per-step rebuild
    /// pattern in the training path keeps the two copies in sync.
    pub fn from_cpu(model: &crate::model::Model) -> Self {
        let blocks = model
            .blocks
            .iter()
            .map(|b| {
                let heads = b
                    .heads
                    .iter()
                    .map(|h| crate::device::HeadWeights {
                        w_q: CudaTensor::from_cpu(&h.w_q).expect("H->D w_q failed"),
                        w_k: CudaTensor::from_cpu(&h.w_k).expect("H->D w_k failed"),
                        w_v: CudaTensor::from_cpu(&h.w_v).expect("H->D w_v failed"),
                        w_o: CudaTensor::from_cpu(&h.w_o).expect("H->D w_o failed"),
                    })
                    .collect();
                let ffn = crate::device::FfnWeights {
                    w_gate: CudaTensor::from_cpu(&b.ffn_gate_w).expect("H->D w_gate failed"),
                    w_up: CudaTensor::from_cpu(&b.ffn_up_w).expect("H->D w_up failed"),
                    w_down: CudaTensor::from_cpu(&b.ffn_down_w).expect("H->D w_down failed"),
                };
                CudaBlockMasters { heads, ffn }
            })
            .collect();
        let token_embed_device =
            CudaTensor::from_cpu(&model.token_embed).expect("H->D token_embed failed");
        Self {
            config: model.config,
            token_embed_cpu: model.token_embed.clone(),
            token_embed_device,
            blocks,
        }
    }

    /// Refresh every device weight in place from the CPU masters
    /// (issue #1). Same walk order as `from_cpu`, but the existing
    /// device buffers are overwritten via `memcpy_htod` instead of
    /// reallocated, so the per-step rebuild allocates zero new device
    /// buffers. Weight shapes are step-invariant during training, so
    /// an architecture mismatch is a caller bug and panics.
    pub fn sync_from_cpu(&mut self, model: &crate::model::Model) {
        assert_eq!(
            self.blocks.len(),
            model.blocks.len(),
            "sync_from_cpu: block count changed"
        );
        for (db, hb) in self.blocks.iter_mut().zip(&model.blocks) {
            assert_eq!(
                db.heads.len(),
                hb.heads.len(),
                "sync_from_cpu: head count changed"
            );
            for (dh, hh) in db.heads.iter_mut().zip(&hb.heads) {
                dh.w_q.copy_from_cpu(&hh.w_q).expect("H->D w_q sync failed");
                dh.w_k.copy_from_cpu(&hh.w_k).expect("H->D w_k sync failed");
                dh.w_v.copy_from_cpu(&hh.w_v).expect("H->D w_v sync failed");
                dh.w_o.copy_from_cpu(&hh.w_o).expect("H->D w_o sync failed");
            }
            db.ffn
                .w_gate
                .copy_from_cpu(&hb.ffn_gate_w)
                .expect("H->D w_gate sync failed");
            db.ffn
                .w_up
                .copy_from_cpu(&hb.ffn_up_w)
                .expect("H->D w_up sync failed");
            db.ffn
                .w_down
                .copy_from_cpu(&hb.ffn_down_w)
                .expect("H->D w_down sync failed");
        }
        // Host-side embed copy: overwrite in place too (shapes match),
        // so the per-step rebuild allocates nothing on either side.
        self.token_embed_cpu
            .data
            .copy_from_slice(&model.token_embed.data);
        self.token_embed_device
            .copy_from_cpu(&model.token_embed)
            .expect("H->D token_embed sync failed");
    }

    /// End-to-end forward pass: token ids -> logits `[seq, vocab]`.
    ///
    /// Pipeline:
    ///   1. CPU embed lookup: row-pick from `token_embed_cpu`.
    ///   2. H->D copy of the gathered `[seq, hidden]` slab.
    ///   3. Generic `block_inference<CudaTensor>` per block, all
    ///      device-resident (no intermediate H<->D bouncing).
    ///   4. Final RMSNorm.
    ///   5. `matmul(token_embed_device.transpose())` -> `[seq, vocab]` logits.
    ///      Tied embeddings since v0.17 / BNT5; the LM-head weight is the
    ///      transposed token-embed table.
    ///
    /// Inference-only path. No BitNet quantisation in Phase 3; weights
    /// are used as their raw f32 masters.
    pub fn forward(&self, ids: &[usize]) -> CudaTensor {
        use crate::device::{MatMul, RmsNorm, Transpose2D, block_inference};
        assert!(
            ids.len() <= self.config.max_seq_len,
            "forward: seq_len {} exceeds max_seq_len {}",
            ids.len(),
            self.config.max_seq_len,
        );
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let table = &self.token_embed_cpu.data;
        // Step 1: CPU row-pick. Allocate the [seq, hidden] slab and
        // fill it from the CPU-side embed table.
        let mut slab: Vec<f32> = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            assert!(id < vocab, "forward: id {id} >= vocab {vocab}");
            slab.extend_from_slice(&table[id * h..(id + 1) * h]);
        }
        let x_cpu = Tensor::from_vec(slab, vec![ids.len(), h]);
        // Step 2: H->D.
        let mut x = CudaTensor::from_cpu(&x_cpu).expect("forward: H->D embed failed");
        // Step 3: blocks, fully device-resident.
        for block in &self.blocks {
            x = block_inference(&x, &block.heads, &block.ffn, self.config.head_dim);
        }
        // Steps 4 + 5. Tied LM head: matmul against the transposed
        // device-resident token_embed.
        let lm_head_w = self.token_embed_device.transpose_2d();
        x.rmsnorm().matmul(&lm_head_w)
    }

    /// Phase 4 chunk 4.5.e: end-to-end forward+backward for one
    /// training window. Returns `(grads, loss)` matching the CPU
    /// `compute_grads_for_window` signature so the existing training
    /// loop can opt in via a single boolean flag.
    ///
    /// **Important caveat**: this path is **f32 throughout** - it does
    /// NOT apply BitNet ternary STE quantisation in either forward or
    /// backward. On a model built from `Model::new(...)` and trained
    /// via this path, the gradient flow goes through full-precision
    /// matmuls, so the resulting weights are f32 (not ternary). Phase 5
    /// will add ternary tensor-core kernels that restore the
    /// BitNet-style training semantics on the GPU. Until then, this
    /// path is useful for f32 ablations, GPU-perf measurements, and
    /// integration tests; it is **not** a drop-in replacement for the
    /// production CPU `compute_grads_for_window` if you want the
    /// ternary-quantised semantics.
    ///
    /// Output gradient ordering matches `Model::for_each_grad`'s
    /// canonical visitor order (v0.17, tied embeddings):
    ///   1. token_embed - accumulates contributions from BOTH the embed
    ///      gather (scatter-add per input id) and the LM-head matmul
    ///      backward (transposed [vocab, hidden] grad slab)
    ///   2. for each block: per-head q/k/v/o, then ffn_gate, ffn_up,
    ///      ffn_down
    ///
    /// (No trailing lm_head: tied to token_embed.)
    pub fn compute_grads_for_window(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
    ) -> (Vec<Tensor>, f32) {
        use crate::device::{
            BlockSaved, MatMul, RmsNorm, RmsNormBackward, Transpose2D, block_backward,
            block_forward_save, cross_entropy_backward, cross_entropy_forward_save,
            matmul_backward,
        };
        assert_eq!(
            input_ids.len(),
            target_ids.len(),
            "compute_grads: input/target length mismatch"
        );
        assert!(
            input_ids.len() <= self.config.max_seq_len,
            "compute_grads: seq_len {} exceeds max_seq_len {}",
            input_ids.len(),
            self.config.max_seq_len,
        );
        let seq = input_ids.len();
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let head_dim = self.config.head_dim;

        // ---- Forward, saving every intermediate the backward needs. ----

        // Step 1: CPU embed gather + H->D.
        let table = &self.token_embed_cpu.data;
        let mut slab: Vec<f32> = Vec::with_capacity(seq * h);
        for &id in input_ids {
            assert!(id < vocab, "compute_grads: id {id} >= vocab {vocab}");
            slab.extend_from_slice(&table[id * h..(id + 1) * h]);
        }
        let x_cpu = Tensor::from_vec(slab, vec![seq, h]);
        let x_post_embed = CudaTensor::from_cpu(&x_cpu).expect("H->D embed");

        // Step 2: walk blocks, accumulating block_inputs[i] = input to
        // block i (so block_inputs[0] = x_post_embed, block_inputs[i+1]
        // = block_forward_save(block_inputs[i]).out).
        let n_blocks = self.blocks.len();
        let mut block_inputs: Vec<CudaTensor> = Vec::with_capacity(n_blocks + 1);
        let mut block_saveds: Vec<BlockSaved<CudaTensor>> = Vec::with_capacity(n_blocks);
        block_inputs.push(x_post_embed);
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, saved) =
                block_forward_save(&block_inputs[i], &block.heads, &block.ffn, head_dim);
            block_saveds.push(saved);
            block_inputs.push(out);
        }
        // final_x = block_inputs[n_blocks]; pre_lm_head = rmsnorm(final_x).
        // Tied LM head: matmul against transposed device-resident token_embed.
        let final_x = &block_inputs[n_blocks];
        let pre_lm_head = final_x.rmsnorm();
        let lm_head_w = self.token_embed_device.transpose_2d(); // [hidden, vocab]
        let logits = pre_lm_head.matmul(&lm_head_w);

        // Step 3: cross-entropy fused softmax + loss.
        let (loss, softmax_saved) = cross_entropy_forward_save(&logits, target_ids);

        // ---- Backward chain. ----

        let grad_logits = cross_entropy_backward(&softmax_saved, target_ids, seq);
        // logits = pre_lm_head @ lm_head_w (= token_embed_device.T)
        let (grad_pre_lm_head, grad_lm_head_w) =
            matmul_backward(&grad_logits, &pre_lm_head, &lm_head_w);
        // grad through the transpose: [hidden, vocab] -> [vocab, hidden].
        // This is the LM-head's contribution to the tied token_embed grad;
        // it gets summed into the embed-gather contribution further down.
        let grad_token_embed_from_lm = grad_lm_head_w
            .transpose_2d()
            .to_cpu()
            .expect("D->H grad_lm");
        // pre_lm_head = rmsnorm(final_x)
        let grad_final_x = grad_pre_lm_head.rmsnorm_backward(final_x);

        // Walk blocks in reverse, accumulating per-block grad bundles.
        let mut grad_x = grad_final_x;
        let mut block_grads_rev: Vec<crate::device::BlockGrads<CudaTensor>> =
            Vec::with_capacity(n_blocks);
        for i in (0..n_blocks).rev() {
            let bg = block_backward(
                &grad_x,
                &block_inputs[i],
                &block_saveds[i],
                &self.blocks[i].heads,
                &self.blocks[i].ffn,
                head_dim,
            );
            // grad_x for the previous-block iteration is this block's
            // grad_x. Clone is one D->D memcpy per block (cheap at
            // v0.13 sizes) and lets the BlockGrads bundle keep
            // ownership of its grad_x for later inspection if needed.
            grad_x = bg.grad_x.clone();
            block_grads_rev.push(bg);
        }
        let block_grads: Vec<crate::device::BlockGrads<CudaTensor>> =
            block_grads_rev.into_iter().rev().collect();

        // Embed backward: scatter-add the gradient flowing into the
        // first block back into a CPU `grad_token_embed` table. We then
        // sum in `grad_token_embed_from_lm` (the LM-head contribution
        // computed above) - tied embeddings means both paths feed into
        // the same parameter slot.
        let grad_x_pre_blocks_cpu = grad_x.to_cpu().expect("D->H grad_x_pre_blocks");
        let mut grad_token_embed = vec![0.0_f32; vocab * h];
        for (pos, &id) in input_ids.iter().enumerate() {
            let dst_start = id * h;
            let src_start = pos * h;
            for c in 0..h {
                grad_token_embed[dst_start + c] += grad_x_pre_blocks_cpu.data[src_start + c];
            }
        }
        debug_assert_eq!(grad_token_embed_from_lm.data.len(), grad_token_embed.len());
        for (g, &lm) in grad_token_embed
            .iter_mut()
            .zip(&grad_token_embed_from_lm.data)
        {
            *g += lm;
        }
        let grad_token_embed_t = Tensor {
            data: grad_token_embed,
            shape: vec![vocab, h],
        };

        // ---- Flatten gradients in canonical visitor order. ----
        // No trailing lm_head: tied to token_embed since v0.17 / BNT5.

        let mut grads: Vec<Tensor> = Vec::new();
        grads.push(grad_token_embed_t);
        for bg in &block_grads {
            for hg in &bg.head_grads {
                grads.push(hg.grad_w_q.to_cpu().expect("D->H w_q"));
                grads.push(hg.grad_w_k.to_cpu().expect("D->H w_k"));
                grads.push(hg.grad_w_v.to_cpu().expect("D->H w_v"));
                grads.push(hg.grad_w_o.to_cpu().expect("D->H w_o"));
            }
            grads.push(bg.grad_w_gate.to_cpu().expect("D->H w_gate"));
            grads.push(bg.grad_w_up.to_cpu().expect("D->H w_up"));
            grads.push(bg.grad_w_down.to_cpu().expect("D->H w_down"));
        }

        (grads, loss)
    }

    /// Phase 5.a: BitNet end-to-end forward+backward on the GPU. Same
    /// pipeline as `compute_grads_for_window` but every learnable-
    /// weight matmul (per-head Q/K/V/O, FFN gate/up/down, lm_head)
    /// goes through `bit_linear` (quantise_acts_ste(x) @
    /// quantise_weights_ste(w)). Matches `Var::cross_entropy(model.forward(...))`
    /// in the autograd CPU training path.
    ///
    /// Returns `(Vec<Tensor>, f32)` in the **canonical visitor order**
    /// matching `Model::for_each_grad`, drop-in compatible with the
    /// existing CPU optimiser.
    ///
    /// Internally this is `alloc_step_buffers` + `step_core_into` +
    /// `assemble_grads`: the same device-only core a captured CUDA
    /// graph replays (issue #3), so the eager path doubles as the
    /// correctness witness for the graph path.
    pub fn compute_grads_for_window_bitnet(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
    ) -> (Vec<Tensor>, f32) {
        let seq = input_ids.len();
        let mut bufs = self.alloc_step_buffers(seq);
        let (x_embed, targets_dev) = self.upload_step_inputs(input_ids, target_ids, None);
        self.step_core_into(&x_embed, &targets_dev, &mut bufs);
        self.assemble_grads(&bufs, input_ids)
    }

    /// Gather the embed slab for `input_ids` on the host and upload it
    /// plus the i32 targets. With `reuse = Some((x_embed, targets))`
    /// the persistent buffers are overwritten in place (graph path);
    /// otherwise fresh device buffers are allocated (eager path).
    fn upload_step_inputs(
        &self,
        input_ids: &[usize],
        target_ids: &[usize],
        reuse: Option<(&mut CudaTensor, &mut CudaSlice<i32>)>,
    ) -> (CudaTensor, CudaSlice<i32>) {
        assert_eq!(
            input_ids.len(),
            target_ids.len(),
            "compute_grads_bitnet: input/target length mismatch"
        );
        assert!(
            input_ids.len() <= self.config.max_seq_len,
            "compute_grads_bitnet: seq_len {} exceeds max_seq_len {}",
            input_ids.len(),
            self.config.max_seq_len,
        );
        let seq = input_ids.len();
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let s = cuda_state().expect("cuda_state failed");

        // CPU embed gather.
        let table = &self.token_embed_cpu.data;
        let mut slab: Vec<f32> = Vec::with_capacity(seq * h);
        for &id in input_ids {
            assert!(id < vocab, "compute_grads_bitnet: id {id} >= vocab {vocab}");
            slab.extend_from_slice(&table[id * h..(id + 1) * h]);
        }
        let x_cpu = Tensor::from_vec(slab, vec![seq, h]);
        let targets_i32: Vec<i32> = target_ids
            .iter()
            .map(|&t| {
                assert!(
                    t < vocab,
                    "compute_grads_bitnet: target {t} >= vocab {vocab}"
                );
                i32::try_from(t).expect("target exceeds i32")
            })
            .collect();

        match reuse {
            Some((x_embed, targets_dev)) => {
                x_embed.copy_from_cpu(&x_cpu).expect("H->D embed sync");
                s.active_stream()
                    .memcpy_htod(&targets_i32, targets_dev)
                    .expect("H->D targets sync");
                (
                    CudaTensor {
                        data: x_embed.data.clone(),
                        shape: x_embed.shape.clone(),
                    },
                    targets_dev.clone(),
                )
            }
            None => {
                let x_embed = CudaTensor::from_cpu(&x_cpu).expect("H->D embed");
                let targets_dev = s
                    .active_stream()
                    .clone_htod(&targets_i32)
                    .expect("H->D targets");
                (x_embed, targets_dev)
            }
        }
    }

    /// Persistent output slot for one training step: ONE contiguous
    /// device buffer holding every output at a precomputed offset
    /// (issue #15), so the whole readback is a single D->H copy.
    /// Allocated once, written by `step_core_into` via device-to-device
    /// copies, read back by `assemble_grads`. The stable address is
    /// also what lets a captured graph (issue #3) deliver its results
    /// across replays.
    ///
    /// Layout (element offsets, all f32):
    ///   [0..seq)              per-row CE loss
    ///   [seq..+vocab*h)       grad of the tied lm_head, pre-transposed
    ///                         to [vocab, hidden]
    ///   [..+seq*h)            grad wrt the post-embed activations
    ///   then per block: per head w_q, w_k, w_v, w_o, then the FFN
    ///   gate, up, down - the same walk `assemble_grads` flattens.
    fn alloc_step_buffers(&self, seq: usize) -> StepBuffers {
        let s = cuda_state().expect("cuda_state failed");
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let mut next = 0usize;
        let mut claim = |len: usize| -> usize {
            let at = next;
            next += len;
            at
        };
        let loss = claim(seq);
        let lm_head_w_t = claim(vocab * h);
        let x_pre_blocks = claim(seq * h);
        let blocks = self
            .blocks
            .iter()
            .map(|b| BlockOffsets {
                heads: b
                    .heads
                    .iter()
                    .map(|hw| {
                        [
                            claim(hw.w_q.data.len()),
                            claim(hw.w_k.data.len()),
                            claim(hw.w_v.data.len()),
                            claim(hw.w_o.data.len()),
                        ]
                    })
                    .collect(),
                gate: claim(b.ffn.w_gate.data.len()),
                up: claim(b.ffn.w_up.data.len()),
                down: claim(b.ffn.w_down.data.len()),
            })
            .collect();
        let total = next;
        StepBuffers {
            flat: s
                .active_stream()
                .alloc_zeros::<f32>(total)
                .expect("alloc flat step buffer"),
            layout: StepLayout {
                loss,
                lm_head_w_t,
                x_pre_blocks,
                blocks,
                total,
            },
        }
    }

    /// Device-to-device copy of one step output into its slot of the
    /// flat buffer. Bounds are guaranteed by construction (`claim`
    /// sized every slot from the same tensors that get written), so a
    /// mismatch is a layout bug and panics.
    fn write_flat(flat: &mut CudaSlice<f32>, offset: usize, src: &CudaSlice<f32>) {
        let s = cuda_state().expect("cuda_state failed");
        let mut dst = flat.slice_mut(offset..offset + src.len());
        s.active_stream()
            .memcpy_dtod(src, &mut dst)
            .expect("D->D into flat step buffer");
    }

    /// The device-only step core: forward through every block, tied
    /// lm_head, fused softmax+CE, then the full backward chain -
    /// finishing with device-to-device copies of every output into the
    /// caller's persistent `StepBuffers`. **No host transfer and no
    /// synchronisation anywhere**, which is exactly what makes this
    /// capturable as a CUDA graph (issue #3): intermediates are
    /// stream-ordered `cuMemAllocAsync`/`cuMemFreeAsync` pairs that
    /// capture as alloc/free nodes, so a replay reuses the same
    /// physical memory step after step.
    fn step_core_into(
        &self,
        x_embed: &CudaTensor,
        targets_dev: &CudaSlice<i32>,
        bufs: &mut StepBuffers,
    ) {
        use crate::device::{
            BitLinear, BlockSaved, RmsNorm, RmsNormBackward, Transpose2D, bit_linear_backward,
            block_backward_bitnet, block_forward_save_bitnet,
        };
        let seq = x_embed.shape[0];
        let head_dim = self.config.head_dim;

        // ---- Forward. ----
        let n_blocks = self.blocks.len();
        let mut block_inputs: Vec<CudaTensor> = Vec::with_capacity(n_blocks + 1);
        let mut block_saveds: Vec<BlockSaved<CudaTensor>> = Vec::with_capacity(n_blocks);
        block_inputs.push(x_embed.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, saved) =
                block_forward_save_bitnet(&block_inputs[i], &block.heads, &block.ffn, head_dim);
            block_saveds.push(saved);
            block_inputs.push(out);
        }
        let final_x = &block_inputs[n_blocks];
        let pre_lm_head = final_x.rmsnorm();
        // Tied LM head: bit_linear over the transposed token_embed device
        // copy. The bit_linear path quantises both the rmsnorm activations
        // (per-row INT8 absmax) and the weight (ternary absmean) on the
        // fly, so the f32 master tensor we feed in is dequantised back to
        // ternary semantics matching the autograd Var path exactly.
        let lm_head_w = self.token_embed_device.transpose_2d(); // [hidden, vocab]
        let logits = pre_lm_head.bit_linear(&lm_head_w);

        // ---- Loss (device-resident; host mean happens in assemble). ----
        let (per_row_loss, softmax_saved) = logits.cross_entropy_forward_device(targets_dev);

        // ---- Backward chain. ----
        let grad_logits = softmax_saved.cross_entropy_backward_device(targets_dev, seq);
        // logits = bit_linear(pre_lm_head, lm_head_w = token_embed_device.T)
        let (grad_pre_lm_head, grad_lm_head_w) =
            bit_linear_backward(&grad_logits, &pre_lm_head, &lm_head_w);
        let grad_final_x = grad_pre_lm_head.rmsnorm_backward(final_x);

        let mut grad_x = grad_final_x;
        let mut block_grads_rev: Vec<crate::device::BlockGrads<CudaTensor>> =
            Vec::with_capacity(n_blocks);
        for i in (0..n_blocks).rev() {
            let bg = block_backward_bitnet(
                &grad_x,
                &block_inputs[i],
                &block_saveds[i],
                &self.blocks[i].heads,
                &self.blocks[i].ffn,
                head_dim,
            );
            grad_x = bg.grad_x.clone();
            block_grads_rev.push(bg);
        }
        let block_grads: Vec<crate::device::BlockGrads<CudaTensor>> =
            block_grads_rev.into_iter().rev().collect();

        // ---- Land every output in its flat-buffer slot (issue #15). ----
        let StepBuffers { flat, layout } = bufs;
        Self::write_flat(flat, layout.loss, &per_row_loss);
        // Tied-embedding contribution transposed to [vocab, hidden] so
        // the host side can sum it straight into the embed-gather grad.
        Self::write_flat(
            flat,
            layout.lm_head_w_t,
            &grad_lm_head_w.transpose_2d().data,
        );
        Self::write_flat(flat, layout.x_pre_blocks, &grad_x.data);
        for (bo, bg) in layout.blocks.iter().zip(&block_grads) {
            for (ho, hg) in bo.heads.iter().zip(&bg.head_grads) {
                Self::write_flat(flat, ho[0], &hg.grad_w_q.data);
                Self::write_flat(flat, ho[1], &hg.grad_w_k.data);
                Self::write_flat(flat, ho[2], &hg.grad_w_v.data);
                Self::write_flat(flat, ho[3], &hg.grad_w_o.data);
            }
            Self::write_flat(flat, bo.gate, &bg.grad_w_gate.data);
            Self::write_flat(flat, bo.up, &bg.grad_w_up.data);
            Self::write_flat(flat, bo.down, &bg.grad_w_down.data);
        }
    }

    /// Pull the step outputs off the device - ONE flat D->H read
    /// (issue #15) - and build the CPU-side result: mean loss, embed
    /// scatter-add + tied-lm_head sum, and the gradient list in the
    /// **canonical visitor order** matching `Model::for_each_grad`.
    /// Shared verbatim by the eager and graph paths, so the two can
    /// only differ in what the device computed - which is what the
    /// graph test pins bitwise.
    fn assemble_grads(&self, bufs: &StepBuffers, input_ids: &[usize]) -> (Vec<Tensor>, f32) {
        let s = cuda_state().expect("cuda_state failed");
        let seq = input_ids.len();
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let lay = &bufs.layout;

        count_dtoh_read();
        let host = s
            .active_stream()
            .clone_dtoh(&bufs.flat)
            .expect("D->H flat step buffer");
        debug_assert_eq!(host.len(), lay.total);
        let at = |offset: usize, len: usize| -> &[f32] { &host[offset..offset + len] };

        let loss = at(lay.loss, seq).iter().sum::<f32>() / seq as f32;

        // Embed scatter-add (CPU side), then sum in the LM-head's
        // tied-embedding contribution.
        let grad_x_pre_blocks = at(lay.x_pre_blocks, seq * h);
        let grad_token_embed_from_lm = at(lay.lm_head_w_t, vocab * h);
        let mut grad_token_embed = vec![0.0_f32; vocab * h];
        for (pos, &id) in input_ids.iter().enumerate() {
            let dst_start = id * h;
            let src_start = pos * h;
            for c in 0..h {
                grad_token_embed[dst_start + c] += grad_x_pre_blocks[src_start + c];
            }
        }
        for (g, &lm) in grad_token_embed.iter_mut().zip(grad_token_embed_from_lm) {
            *g += lm;
        }
        let grad_token_embed_t = Tensor {
            data: grad_token_embed,
            shape: vec![vocab, h],
        };

        // Flatten in canonical visitor order, slicing each parameter's
        // gradient out of the flat host copy at its layout offset.
        // Shapes come from the device weight tensors - the same source
        // `alloc_step_buffers` sized the slots from. No trailing
        // lm_head: tied to token_embed since v0.17 / BNT5.
        let slice_to_tensor = |offset: usize, shape: &[usize]| -> Tensor {
            let len = shape.iter().product();
            Tensor {
                data: at(offset, len).to_vec(),
                shape: shape.to_vec(),
            }
        };
        let mut grads: Vec<Tensor> = Vec::new();
        grads.push(grad_token_embed_t);
        for (bo, b) in lay.blocks.iter().zip(&self.blocks) {
            for (ho, hw) in bo.heads.iter().zip(&b.heads) {
                grads.push(slice_to_tensor(ho[0], &hw.w_q.shape));
                grads.push(slice_to_tensor(ho[1], &hw.w_k.shape));
                grads.push(slice_to_tensor(ho[2], &hw.w_v.shape));
                grads.push(slice_to_tensor(ho[3], &hw.w_o.shape));
            }
            grads.push(slice_to_tensor(bo.gate, &b.ffn.w_gate.shape));
            grads.push(slice_to_tensor(bo.up, &b.ffn.w_up.shape));
            grads.push(slice_to_tensor(bo.down, &b.ffn.w_down.shape));
        }

        (grads, loss)
    }

    /// Capture the whole device-side training step as a CUDA graph
    /// (issue #3). One eager warm-up run primes the NVRTC modules,
    /// the cuBLAS int8 workspace, and the stream-ordered memory pool
    /// (none of which may first-allocate mid-capture); the second run
    /// is recorded on a **private stream** with capture mode
    /// `THREAD_LOCAL`, so concurrent CUDA work from other threads can
    /// neither leak into the graph nor be blocked by it.
    ///
    /// Validity across steps: the graph references every weight
    /// buffer by address, and `sync_from_cpu` (issue #1) refreshes
    /// those buffers strictly in place - so a replay always reads the
    /// current weights. The persistent input/output buffers live in
    /// the returned `CudaStepGraph`.
    pub fn build_step_graph(&self, seq: usize) -> Result<CudaStepGraph, String> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        let s = cuda_state()?;
        // Cross-stream edge: weight buffers are written on the default
        // stream (from_cpu / sync_from_cpu). Drain it before the
        // private stream starts reading them.
        s.stream
            .synchronize()
            .map_err(|e| format!("default-stream sync failed: {e:?}"))?;
        let capture_stream = s
            .ctx
            .new_stream()
            .map_err(|e| format!("new_stream failed: {e:?}"))?;
        let capture_blas = Arc::new(
            CudaBlas::new(capture_stream.clone())
                .map_err(|e| format!("CudaBlas::new (capture) failed: {e:?}"))?,
        );
        let _guard = StreamOverrideGuard::install(capture_stream.clone(), capture_blas.clone());

        let h = self.config.hidden_dim;
        let x_embed = CudaTensor {
            data: capture_stream
                .alloc_zeros::<f32>(seq * h)
                .map_err(|e| format!("alloc x_embed failed: {e:?}"))?,
            shape: vec![seq, h],
        };
        let targets_dev = capture_stream
            .alloc_zeros::<i32>(seq)
            .map_err(|e| format!("alloc targets failed: {e:?}"))?;
        let mut bufs = self.alloc_step_buffers(seq);

        // Warm-up (eager, on the capture stream).
        self.step_core_into(&x_embed, &targets_dev, &mut bufs);
        capture_stream
            .synchronize()
            .map_err(|e| format!("warm-up sync failed: {e:?}"))?;

        // Recorded run.
        capture_stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| format!("begin_capture failed: {e:?}"))?;
        self.step_core_into(&x_embed, &targets_dev, &mut bufs);
        let graph = capture_stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| format!("end_capture failed: {e:?}"))?
            .ok_or_else(|| "end_capture returned no graph".to_string())?;
        graph
            .upload()
            .map_err(|e| format!("graph upload failed: {e:?}"))?;

        Ok(CudaStepGraph {
            graph,
            stream: capture_stream,
            blas: capture_blas,
            x_embed,
            targets_dev,
            bufs,
            seq,
        })
    }

    /// Replay the captured step for one window: overwrite the
    /// persistent input buffers, launch the graph (ONE driver call for
    /// the whole forward+backward), then assemble the grads on the
    /// host. Weight freshness is the caller's business via
    /// `sync_from_cpu`, same as the eager path.
    pub fn run_step_graph(
        &self,
        g: &mut CudaStepGraph,
        input_ids: &[usize],
        target_ids: &[usize],
    ) -> (Vec<Tensor>, f32) {
        assert_eq!(
            input_ids.len(),
            g.seq,
            "run_step_graph: seq {} does not match captured seq {}",
            input_ids.len(),
            g.seq,
        );
        // Cross-stream edge: `sync_from_cpu` uploads the fresh weights
        // on the default stream; drain it so the replay reads them.
        cuda_state()
            .expect("cuda_state failed")
            .stream
            .synchronize()
            .expect("default-stream sync failed");
        // Everything below must ride the graph's private stream: the
        // input htod precedes the launch, the assemble dtoh follows it,
        // and stream order is what sequences them correctly.
        let _guard = StreamOverrideGuard::install(g.stream.clone(), g.blas.clone());
        let _ = self.upload_step_inputs(
            input_ids,
            target_ids,
            Some((&mut g.x_embed, &mut g.targets_dev)),
        );
        g.graph.launch().expect("graph launch failed");
        self.assemble_grads(&g.bufs, input_ids)
    }
}

/// Persistent per-step output buffer (issues #3 + #15): one flat
/// device allocation plus the element offsets of every slot in it.
/// See `CudaModel::alloc_step_buffers` for the layout.
struct StepBuffers {
    flat: CudaSlice<f32>,
    layout: StepLayout,
}

/// Element offsets into `StepBuffers::flat`. Shapes are not stored:
/// both the writer (`step_core_into`) and the reader
/// (`assemble_grads`) take them from the model's device weight
/// tensors, the same source `alloc_step_buffers` sized the slots from.
struct StepLayout {
    loss: usize,
    lm_head_w_t: usize,
    x_pre_blocks: usize,
    blocks: Vec<BlockOffsets>,
    total: usize,
}

/// Per-block slot offsets: `heads[i]` holds `[w_q, w_k, w_v, w_o]`.
struct BlockOffsets {
    heads: Vec<[usize; 4]>,
    gate: usize,
    up: usize,
    down: usize,
}

/// A captured training step: the instantiated CUDA graph plus every
/// buffer it references that must outlive and feed each replay. Built
/// by `CudaModel::build_step_graph`, driven by `run_step_graph`.
pub struct CudaStepGraph {
    graph: cudarc::driver::CudaGraph,
    /// The private stream the graph was captured on; replays and their
    /// surrounding H<->D copies are ordered on it.
    stream: Arc<CudaStream>,
    /// cuBLAS handle bound to `stream` - kept alive because the
    /// captured GEMM nodes were recorded through it.
    #[allow(dead_code)]
    blas: Arc<CudaBlas>,
    x_embed: CudaTensor,
    targets_dev: CudaSlice<i32>,
    bufs: StepBuffers,
    seq: usize,
}

/// Convenience: copy two CPU tensors to device, multiply, copy back.
/// Per-call H<->D overhead dominates for the matmul shapes the v0.13
/// model produces (~150 us for the copies vs ~10 us for the kernel), so
/// this is intentionally a demo / test entry point and not used by
/// training. Phase 2 will keep tensors device-resident across whole
/// blocks.
pub fn cuda_matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, String> {
    use crate::device::MatMul;
    let l = CudaTensor::from_cpu(lhs)?;
    let r = CudaTensor::from_cpu(rhs)?;
    l.matmul(&r).to_cpu()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{
        Add, CausalMask, MatMul, Mul, MulScalar, RmsNorm, Rope, Silu, Softmax, Transpose2D,
    };

    /// Maximum acceptable absolute error per cell from a CPU-vs-CUDA
    /// matmul. Parallel reduction across thread blocks reorders the
    /// per-cell sum, and f32 add is non-associative, so bit-equality is
    /// impossible for any non-trivial K. The bound below is loose enough
    /// to absorb that drift across the prime-shaped test matrices.
    const ABS_TOL: f32 = 1e-4;
    const REL_TOL: f32 = 1e-4;

    fn assert_close(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape, b.shape, "shape mismatch");
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (&x, &y)) in a.data.iter().zip(&b.data).enumerate() {
            let abs = (x - y).abs();
            let rel = if y.abs() > 1e-6 { abs / y.abs() } else { 0.0 };
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            assert!(
                abs <= ABS_TOL + REL_TOL * y.abs(),
                "drift at idx {i}: cpu = {y}, cuda = {x}, |diff| = {abs}"
            );
        }
        eprintln!("cpu vs cuda: max |diff| = {max_abs:.3e}, max rel = {max_rel:.3e}");
    }

    /// Round-trip through device memory must preserve the bytes exactly:
    /// no math, just memcpy, so the tolerance here is byte-equality.
    #[test]
    fn cuda_round_trip_preserves_bytes() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let host = Tensor::from_vec((0..30).map(|i| i as f32 * 0.123).collect(), vec![5, 6]);
        let dev = CudaTensor::from_cpu(&host).expect("H->D copy failed");
        let back = dev.to_cpu().expect("D->H copy failed");
        assert_eq!(host.shape, back.shape);
        assert_eq!(host.data, back.data, "round-trip lost bytes");
    }

    /// Tile-aligned 32x32 @ 32x32 - all four edges land on TILE
    /// boundaries, so no boundary-mask thread does anything different
    /// from an interior thread. Should agree with CPU within the
    /// floating-point tolerance defined above.
    #[test]
    fn cuda_matmul_tile_aligned_matches_cpu() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 32usize;
        let k = 32usize;
        let n = 32usize;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        let cpu = lhs.matmul(&rhs);
        let gpu = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");
        assert_close(&gpu, &cpu);
    }

    /// Prime dimensions push every output edge through the boundary
    /// guard inside the kernel (`row < m && col < n`) and also exercise
    /// the partial-tile K-loop. If the bounds check is wrong - off by
    /// one, transposed, etc. - this test will diverge wildly, not just
    /// in the last bit.
    #[test]
    fn cuda_matmul_non_tile_aligned_matches_cpu() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 17usize;
        let k = 23usize;
        let n = 31usize;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        let cpu = lhs.matmul(&rhs);
        let gpu = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");
        assert_close(&gpu, &cpu);
    }

    /// cuBLAS sgemm and the hand-rolled NVRTC tile kernel both compute
    /// the same row-major GEMM, but their per-cell summation orders
    /// differ (cuBLAS uses internal blocking optimised for tensor cores
    /// / SM scheduling; the NVRTC path uses a fixed 16x16 tile order).
    /// f32 is non-associative, so they will diverge by a few ULPs. The
    /// tolerance below is the same as cpu-vs-cuda, which is loose enough
    /// to absorb both backends' differences from the textbook ordering.
    #[test]
    fn cublas_and_nvrtc_kernels_agree_within_tolerance() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 64usize;
        let k = 192usize;
        let n = 384usize;
        let lhs_data: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.0173).sin() * 0.5)
            .collect();
        let rhs_data: Vec<f32> = (0..k * n)
            .map(|i| (i as f32 * 0.0259).cos() * 0.1)
            .collect();
        let lhs =
            CudaTensor::from_cpu(&Tensor::from_vec(lhs_data, vec![m, k])).expect("H->D failed");
        let rhs =
            CudaTensor::from_cpu(&Tensor::from_vec(rhs_data, vec![k, n])).expect("H->D failed");

        let cublas_out = lhs.matmul(&rhs).to_cpu().unwrap();
        let nvrtc_out = lhs.matmul_nvrtc(&rhs).to_cpu().unwrap();
        assert_close(&cublas_out, &nvrtc_out);
    }

    /// Realistic v0.13 attention shape: hidden_dim 192 broken into
    /// 12 heads x 16 head_dim. Tests Q computation: x [seq=64, hidden=192]
    /// times W_q [hidden=192, head_dim=16] -> [seq=64, head_dim=16]. The
    /// rectangular and skewed shapes catch grid-dim arithmetic bugs that
    /// the square test would not.
    #[test]
    fn cuda_matmul_attention_q_shape_matches_cpu() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let seq = 64usize;
        let hidden = 192usize;
        let head_dim = 16usize;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.0173).sin() * 0.5)
            .collect();
        let w_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.0259).cos() * 0.1)
            .collect();
        let x = Tensor::from_vec(x_data, vec![seq, hidden]);
        let w = Tensor::from_vec(w_data, vec![hidden, head_dim]);

        let cpu = x.matmul(&w);
        let gpu = cuda_matmul(&x, &w).expect("CUDA matmul failed");
        assert_close(&gpu, &cpu);
    }

    // ---- Phase 2.2 per-op CPU vs CUDA agreement tests. Each one
    // computes the op on a CPU `Tensor`, copies the same input to a
    // `CudaTensor`, runs the trait-method GPU path, and asserts the
    // results agree within tolerance. Uses prime-ish dimensions where
    // possible so boundary handling is exercised in addition to the
    // happy interior path.

    fn random_tensor(rows: usize, cols: usize, seed: f32) -> Tensor {
        Tensor::from_vec(
            (0..rows * cols)
                .map(|i| (i as f32 * 0.0173 + seed).sin() * 0.5)
                .collect(),
            vec![rows, cols],
        )
    }

    /// **Preflight**. Every other CUDA test starts with
    /// `if cuda_state().is_err() { skip }` so the suite stays green on
    /// machines without a GPU. That defensive skip masked a real bug
    /// in Chunk 2.2 onwards: `INFINITY` is undefined under NVRTC, so
    /// `cuda_state()` failed during NVRTC compile, every cross-backend
    /// test silently returned-as-pass, and the GPU code was never
    /// actually exercised.
    ///
    /// This preflight breaks that pattern: on a machine where
    /// `nvidia-smi` reports a usable GPU (the `EXPECT_CUDA` env var)
    /// `cuda_state()` MUST succeed. The skip-on-error pattern in the
    /// other tests stays for cross-machine portability, but this one
    /// fails loudly if the kernels themselves don't compile.
    #[test]
    fn preflight_cuda_state_initialises_cleanly_when_expected() {
        let expect = std::env::var("EXPECT_CUDA").ok();
        match cuda_state() {
            Ok(_) => {}
            Err(e) if expect.is_none() => {
                // No GPU + no expectation that there should be one.
                eprintln!("skipping: no usable CUDA device ({e})");
            }
            Err(e) => {
                panic!("EXPECT_CUDA was set but cuda_state failed: {e}");
            }
        }
    }

    #[test]
    fn add_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(7, 19, 0.1);
        let b = random_tensor(7, 19, 0.7);
        let cpu = <Tensor as Add>::add(&a, &b);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .add(&CudaTensor::from_cpu(&b).unwrap())
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn mul_scalar_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(11, 23, 0.2);
        let cpu = <Tensor as MulScalar>::mul_scalar(&a, 0.31415);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .mul_scalar(0.31415)
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn transpose_2d_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(13, 17, 0.3);
        let cpu = <Tensor as Transpose2D>::transpose_2d(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .transpose_2d()
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn causal_mask_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Square seq x seq is the only shape causal_mask is meaningful
        // on; primes (17) push every edge through the bounds checks.
        let a = random_tensor(17, 17, 0.4);
        let cpu = <Tensor as CausalMask>::causal_mask(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .causal_mask()
            .to_cpu()
            .unwrap();
        // -inf cells must match exactly. Comparing -inf with assert_close's
        // f32 tolerance produces NaN diff; do a separate equality pass.
        for (i, (&c, &g)) in cpu.data.iter().zip(&gpu.data).enumerate() {
            if c.is_infinite() {
                assert_eq!(c, g, "infinity mismatch at idx {i}");
            } else {
                let abs = (c - g).abs();
                assert!(
                    abs <= ABS_TOL + REL_TOL * c.abs(),
                    "drift at idx {i}: cpu = {c}, cuda = {g}"
                );
            }
        }
    }

    #[test]
    fn softmax_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(7, 31, 0.5);
        let cpu = <Tensor as Softmax>::softmax(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .softmax()
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
        // Each GPU row must sum to ~1 - the headline correctness
        // property of softmax. Tighter tolerance than the per-cell
        // diff because sum-of-row is an aggregate.
        for i in 0..gpu.shape[0] {
            let s: f32 = gpu.data[i * gpu.shape[1]..(i + 1) * gpu.shape[1]]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-5, "GPU row {i} sum = {s}");
        }
    }

    #[test]
    fn rope_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // head_dim must be even. Use realistic v0.13 attention shape.
        let a = random_tensor(64, 16, 0.6);
        let cpu = <Tensor as Rope>::rope(&a);
        let gpu = CudaTensor::from_cpu(&a).unwrap().rope().to_cpu().unwrap();
        // RoPE uses transcendentals (cos/sin/pow); GPU + CPU may use
        // slightly different math implementations. Slightly looser
        // tolerance than the elementwise-arithmetic ops.
        for (i, (&c, &g)) in cpu.data.iter().zip(&gpu.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "rope drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    #[test]
    fn silu_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(13, 19, 0.7);
        let cpu = <Tensor as Silu>::silu(&a);
        let gpu = CudaTensor::from_cpu(&a).unwrap().silu().to_cpu().unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn mul_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(11, 17, 0.8);
        let b = random_tensor(11, 17, 0.9);
        let cpu = <Tensor as Mul>::mul(&a, &b);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .mul(&CudaTensor::from_cpu(&b).unwrap())
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn rmsnorm_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(13, 31, 1.5);
        let cpu = <Tensor as RmsNorm>::rmsnorm(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .rmsnorm()
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
        // Each row must end up at unit RMS magnitude (within tolerance):
        // sum(y^2) / n must equal 1 + EPS_norm. The headline correctness
        // property of RMSNorm.
        let n = gpu.shape[1];
        let n_f = n as f32;
        for i in 0..gpu.shape[0] {
            let row = &gpu.data[i * n..(i + 1) * n];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / n_f;
            assert!(
                (mean_sq - 1.0).abs() < 1e-3,
                "GPU row {i} mean_sq = {mean_sq} (expected ~1)"
            );
        }
    }

    /// CPU end-to-end forward built from the same generic helpers
    /// the GPU `CudaModel::forward` uses. Test-only; production CPU
    /// training stays on the existing autograd `Model::forward`. This
    /// helper is the apples-to-apples reference for the cross-backend
    /// model test below.
    fn cpu_forward_unquantised(model: &crate::model::Model, ids: &[usize]) -> Tensor {
        use crate::device::{FfnWeights, HeadWeights, block_inference};
        let h = model.config.hidden_dim;
        let vocab = model.config.vocab_size;
        let table = &model.token_embed.data;
        let mut slab: Vec<f32> = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            assert!(id < vocab, "embed id out of range");
            slab.extend_from_slice(&table[id * h..(id + 1) * h]);
        }
        let mut x = Tensor::from_vec(slab, vec![ids.len(), h]);
        for b in &model.blocks {
            let heads: Vec<HeadWeights<Tensor>> = b
                .heads
                .iter()
                .map(|h| HeadWeights {
                    w_q: h.w_q.clone(),
                    w_k: h.w_k.clone(),
                    w_v: h.w_v.clone(),
                    w_o: h.w_o.clone(),
                })
                .collect();
            let ffn = FfnWeights {
                w_gate: b.ffn_gate_w.clone(),
                w_up: b.ffn_up_w.clone(),
                w_down: b.ffn_down_w.clone(),
            };
            x = block_inference(&x, &heads, &ffn, model.config.head_dim);
        }
        x.rmsnorm().matmul(&model.token_embed.transpose_2d())
    }

    /// **The headline test of Phase 3.** End-to-end model forward
    /// runs on the GPU through `CudaModel::forward` and matches the
    /// equivalent CPU forward (built from the same generic helpers)
    /// within a tolerance proportional to the depth of the chained
    /// op stack. Validates that:
    /// - `CudaModel::from_cpu` correctly mirrors every weight tensor
    /// - `CudaModel::forward` correctly chains embed -> blocks ->
    ///   rmsnorm -> lm_head matmul
    /// - the trait surface composes cleanly across two transformer
    ///   blocks worth of ops
    #[test]
    fn cuda_model_forward_matches_cpu_block_inference_chain() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        let config = ModelConfig {
            vocab_size: 17,
            hidden_dim: 32,
            n_heads: 4,
            head_dim: 8,
            ffn_dim: 64,
            max_seq_len: 16,
            n_blocks: 2,
        };
        let model = Model::new(&config, 42);
        let ids: Vec<usize> = vec![0, 3, 7, 11, 5, 1, 9];

        let cpu_logits = cpu_forward_unquantised(&model, &ids);
        let cuda_model = CudaModel::from_cpu(&model);
        let gpu_logits = cuda_model.forward(&ids).to_cpu().unwrap();

        assert_eq!(cpu_logits.shape, gpu_logits.shape, "logits shape mismatch");
        // Tolerance: each block adds ~5e-3 of drift on a single
        // forward; lm_head matmul adds another small chunk. 2 blocks +
        // final matmul: budget ~2e-2 absolute. Loosened from the
        // single-block test for chain length.
        for (i, (&c, &g)) in cpu_logits.data.iter().zip(&gpu_logits.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 2e-2 + 2e-2 * c.abs(),
                "model logits drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
        // Sanity: last-token argmax from both backends must match -
        // sampling decisions should be identical even if intermediate
        // logits differ at the last few mantissa bits.
        let last_offset = (ids.len() - 1) * config.vocab_size;
        let cpu_argmax = (0..config.vocab_size)
            .max_by(|&a, &b| {
                cpu_logits.data[last_offset + a]
                    .partial_cmp(&cpu_logits.data[last_offset + b])
                    .unwrap()
            })
            .unwrap();
        let gpu_argmax = (0..config.vocab_size)
            .max_by(|&a, &b| {
                gpu_logits.data[last_offset + a]
                    .partial_cmp(&gpu_logits.data[last_offset + b])
                    .unwrap()
            })
            .unwrap();
        assert_eq!(cpu_argmax, gpu_argmax, "greedy-token disagreement");
    }

    /// Issue #3: a captured training-step graph must replay the exact
    /// eager computation. First replay: bitwise-equal grads + loss for
    /// the same window (same kernels, same order, same values). Then
    /// the weights move (simulated optimiser step, synced in place via
    /// `sync_from_cpu`) and a different window is fed: the replay must
    /// see both the refreshed weights and the new inputs, because the
    /// graph references the persistent buffers by address.
    #[test]
    fn cuda_step_graph_replay_matches_eager() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        // Every matmul dimension a multiple of 4 so the int8 GEMM path
        // (not the f32 fallback) is what gets captured.
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 32,
            n_heads: 4,
            head_dim: 8,
            ffn_dim: 64,
            max_seq_len: 8,
            n_blocks: 2,
        };
        let mut model = Model::new(&config, 7);
        let mut cuda_model = CudaModel::from_cpu(&model);
        let win_a: Vec<usize> = vec![0, 3, 7, 11, 5, 1, 9, 2];
        let tgt_a: Vec<usize> = vec![3, 7, 11, 5, 1, 9, 2, 0];

        let (eager_grads, eager_loss) = cuda_model.compute_grads_for_window_bitnet(&win_a, &tgt_a);

        let mut graph = cuda_model
            .build_step_graph(win_a.len())
            .expect("graph capture failed");
        let (graph_grads, graph_loss) = cuda_model.run_step_graph(&mut graph, &win_a, &tgt_a);
        assert_eq!(eager_loss, graph_loss, "loss diverged on first replay");
        assert_eq!(eager_grads.len(), graph_grads.len());
        for (i, (e, g)) in eager_grads.iter().zip(&graph_grads).enumerate() {
            assert_eq!(e.data, g.data, "grad {i} diverged on first replay");
        }

        // Simulated optimiser step + fresh window.
        model.for_each_param_mut(|t| {
            for v in t.data.iter_mut() {
                *v *= 0.9;
            }
        });
        cuda_model.sync_from_cpu(&model);
        let win_b: Vec<usize> = vec![5, 5, 2, 8, 1, 0, 4, 6];
        let tgt_b: Vec<usize> = vec![5, 2, 8, 1, 0, 4, 6, 3];
        let (eager_b, loss_b) = cuda_model.compute_grads_for_window_bitnet(&win_b, &tgt_b);
        let (graph_b, graph_loss_b) = cuda_model.run_step_graph(&mut graph, &win_b, &tgt_b);
        assert_eq!(loss_b, graph_loss_b, "loss diverged after weight sync");
        for (i, (e, g)) in eager_b.iter().zip(&graph_b).enumerate() {
            assert_eq!(e.data, g.data, "grad {i} diverged after weight sync");
        }
    }

    /// Issue #15: the whole step readback must be ONE device-to-host
    /// copy - a single flat gradient buffer read - on both the eager
    /// and the graph path. ~300 per-tensor reads per window were the
    /// dominant residual GPU step cost after #1-#3.
    #[test]
    fn step_readback_is_single_dtoh() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 32,
            n_heads: 4,
            head_dim: 8,
            ffn_dim: 64,
            max_seq_len: 8,
            n_blocks: 2,
        };
        let model = Model::new(&config, 11);
        let cuda_model = CudaModel::from_cpu(&model);
        let win: Vec<usize> = vec![0, 3, 7, 11, 5, 1, 9, 2];
        let tgt: Vec<usize> = vec![3, 7, 11, 5, 1, 9, 2, 0];

        let before = DTOH_READS.with(std::cell::Cell::get);
        let _ = cuda_model.compute_grads_for_window_bitnet(&win, &tgt);
        let eager_reads = DTOH_READS.with(std::cell::Cell::get) - before;
        assert_eq!(
            eager_reads, 1,
            "eager step must read back one flat buffer, saw {eager_reads} D->H reads"
        );

        let mut graph = cuda_model.build_step_graph(8).expect("capture failed");
        let before = DTOH_READS.with(std::cell::Cell::get);
        let _ = cuda_model.run_step_graph(&mut graph, &win, &tgt);
        let graph_reads = DTOH_READS.with(std::cell::Cell::get) - before;
        assert_eq!(
            graph_reads, 1,
            "graph replay must read back one flat buffer, saw {graph_reads} D->H reads"
        );
    }

    /// Issue #3 DoD: the loss trajectory of a short training run must
    /// be identical between the eager path and the graph replay path.
    /// Each step computes grads both ways on the same weights, asserts
    /// bitwise agreement, then applies a plain SGD update and syncs
    /// the device copy in place - so any divergence compounds and
    /// cannot hide.
    #[test]
    fn cuda_step_graph_loss_trajectory_matches_eager() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 32,
            n_heads: 4,
            head_dim: 8,
            ffn_dim: 64,
            max_seq_len: 8,
            n_blocks: 2,
        };
        let mut model = Model::new(&config, 21);
        let mut cuda_model = CudaModel::from_cpu(&model);
        let mut graph = cuda_model
            .build_step_graph(8)
            .expect("graph capture failed");
        let windows: [(Vec<usize>, Vec<usize>); 2] = [
            (vec![0, 3, 7, 11, 5, 1, 9, 2], vec![3, 7, 11, 5, 1, 9, 2, 0]),
            (vec![5, 5, 2, 8, 1, 0, 4, 6], vec![5, 2, 8, 1, 0, 4, 6, 3]),
        ];
        let mut losses = Vec::new();
        for step in 0..6 {
            let (input, target) = &windows[step % windows.len()];
            let (eager_grads, eager_loss) =
                cuda_model.compute_grads_for_window_bitnet(input, target);
            let (graph_grads, graph_loss) = cuda_model.run_step_graph(&mut graph, input, target);
            assert_eq!(eager_loss, graph_loss, "loss diverged at step {step}");
            for (i, (e, g)) in eager_grads.iter().zip(&graph_grads).enumerate() {
                assert_eq!(e.data, g.data, "grad {i} diverged at step {step}");
            }
            losses.push(eager_loss);
            // Plain SGD step on the CPU masters, then in-place device sync.
            let mut i = 0usize;
            model.for_each_param_mut(|t| {
                for (v, g) in t.data.iter_mut().zip(&eager_grads[i].data) {
                    *v -= 0.05 * g;
                }
                i += 1;
            });
            cuda_model.sync_from_cpu(&model);
        }
        // Sanity: training actually trained (first window's loss fell).
        assert!(
            losses[4] < losses[0],
            "loss did not decrease over the trajectory: {losses:?}"
        );
    }

    /// Issue #3 wall-clock bench: eager per-kernel launches vs one
    /// graph replay per window, at the v0.13 shakespeare shape. Not a
    /// correctness gate (timing-dependent), so ignored by default:
    /// `EXPECT_CUDA=1 cargo test --release --features cuda step_graph_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark, run explicitly with --ignored --nocapture"]
    fn cuda_step_graph_bench_vs_eager() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        // v0.13 shakespeare shape (vocab 65 exercises the lm_head f32
        // fallback inside the capture, like the real run).
        let config = ModelConfig {
            vocab_size: 65,
            hidden_dim: 192,
            n_heads: 12,
            head_dim: 16,
            ffn_dim: 384,
            max_seq_len: 64,
            n_blocks: 6,
        };
        let model = Model::new(&config, 3);
        let cuda_model = CudaModel::from_cpu(&model);
        let input: Vec<usize> = (0..64).map(|i| (i * 7) % 65).collect();
        let target: Vec<usize> = (0..64).map(|i| (i * 7 + 1) % 65).collect();
        let iters = 50usize;

        // Warm both paths before timing.
        let _ = cuda_model.compute_grads_for_window_bitnet(&input, &target);
        let mut graph = cuda_model
            .build_step_graph(64)
            .expect("graph capture failed");
        let _ = cuda_model.run_step_graph(&mut graph, &input, &target);

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = cuda_model.compute_grads_for_window_bitnet(&input, &target);
        }
        let eager = t0.elapsed();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = cuda_model.run_step_graph(&mut graph, &input, &target);
        }
        let replay = t0.elapsed();

        println!(
            "per-window forward+backward, v0.13 shape, {iters} iters:\n  eager:  {:>8.2} ms/window\n  graph:  {:>8.2} ms/window\n  speedup: x{:.2}",
            eager.as_secs_f64() * 1e3 / iters as f64,
            replay.as_secs_f64() * 1e3 / iters as f64,
            eager.as_secs_f64() / replay.as_secs_f64(),
        );
    }

    /// Issue #1: `sync_from_cpu` must refresh the device weights in
    /// place - same device buffers (zero new allocations), new values.
    /// Bitwise agreement with a freshly built `from_cpu` model proves
    /// the copy is complete; device-pointer equality proves the
    /// existing allocations were reused rather than replaced.
    #[test]
    fn cuda_model_sync_from_cpu_reuses_buffers_and_matches_rebuild() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        use cudarc::driver::DevicePtr;
        let config = ModelConfig {
            vocab_size: 17,
            hidden_dim: 32,
            n_heads: 4,
            head_dim: 8,
            ffn_dim: 64,
            max_seq_len: 16,
            n_blocks: 2,
        };
        let mut model = Model::new(&config, 42);
        let ids: Vec<usize> = vec![0, 3, 7, 11, 5, 1, 9];
        let mut cuda_model = CudaModel::from_cpu(&model);
        let stale_logits = cuda_model.forward(&ids).to_cpu().unwrap();

        let st = cuda_state().unwrap();
        let w_q_ptr_before = cuda_model.blocks[0].heads[0]
            .w_q
            .data
            .device_ptr(&st.stream)
            .0;
        let embed_ptr_before = cuda_model.token_embed_device.data.device_ptr(&st.stream).0;

        // Simulate an optimiser step: every master weight moves.
        model.for_each_param_mut(|t| {
            for v in t.data.iter_mut() {
                *v += 0.25;
            }
        });

        cuda_model.sync_from_cpu(&model);

        let w_q_ptr_after = cuda_model.blocks[0].heads[0]
            .w_q
            .data
            .device_ptr(&st.stream)
            .0;
        let embed_ptr_after = cuda_model.token_embed_device.data.device_ptr(&st.stream).0;
        assert_eq!(
            w_q_ptr_before, w_q_ptr_after,
            "w_q device buffer was reallocated"
        );
        assert_eq!(
            embed_ptr_before, embed_ptr_after,
            "token_embed device buffer was reallocated"
        );

        let synced = cuda_model.forward(&ids).to_cpu().unwrap();
        let rebuilt = CudaModel::from_cpu(&model).forward(&ids).to_cpu().unwrap();
        assert_eq!(
            synced.data, rebuilt.data,
            "synced forward differs from a fresh from_cpu rebuild"
        );
        assert_ne!(
            synced.data, stale_logits.data,
            "sync_from_cpu left the device weights stale"
        );
    }

    /// **The headline test of Chunk 2.4.** Full transformer block
    /// (pre-norm, multi-head attention with residual, FFN with
    /// residual) runs on both backends from the same source line and
    /// the outputs agree within tolerance. This is also the smallest
    /// test that exercises every Phase 2 trait at least once.
    #[test]
    fn block_inference_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::{FfnWeights, HeadWeights, block_inference};
        let seq = 16usize;
        let n_heads = 3usize;
        let head_dim = 8usize;
        let hidden = n_heads * head_dim;
        let ffn = 64usize;

        // Build identical weights on both backends from the same source
        // tensors. Per-head offset prevents two heads being numerically
        // identical (same trick the autograd-side block test uses).
        let x_cpu = random_tensor(seq, hidden, 3.0);
        let cpu_heads: Vec<HeadWeights<Tensor>> = (0..n_heads)
            .map(|h| {
                let off = 0.001 * (h as f32);
                HeadWeights {
                    w_q: random_tensor(hidden, head_dim, 3.1 + off),
                    w_k: random_tensor(hidden, head_dim, 3.2 + off),
                    w_v: random_tensor(hidden, head_dim, 3.3 + off),
                    w_o: random_tensor(head_dim, hidden, 3.4 + off),
                }
            })
            .collect();
        let cpu_ffn = FfnWeights {
            w_gate: random_tensor(hidden, ffn, 3.5),
            w_up: random_tensor(hidden, ffn, 3.6),
            w_down: random_tensor(ffn, hidden, 3.7),
        };
        let cpu_out = block_inference::<Tensor>(&x_cpu, &cpu_heads, &cpu_ffn, head_dim);

        // Same source, on the GPU. Take ownership of CPU tensors when
        // copying since the helper signature is `block_inference<T>(&T,
        // &[HeadWeights<T>], ...)` and we need a `Vec<HeadWeights<CudaTensor>>`.
        let x_gpu = CudaTensor::from_cpu(&x_cpu).unwrap();
        let gpu_heads: Vec<HeadWeights<CudaTensor>> = cpu_heads
            .iter()
            .map(|h| HeadWeights {
                w_q: CudaTensor::from_cpu(&h.w_q).unwrap(),
                w_k: CudaTensor::from_cpu(&h.w_k).unwrap(),
                w_v: CudaTensor::from_cpu(&h.w_v).unwrap(),
                w_o: CudaTensor::from_cpu(&h.w_o).unwrap(),
            })
            .collect();
        let gpu_ffn = FfnWeights {
            w_gate: CudaTensor::from_cpu(&cpu_ffn.w_gate).unwrap(),
            w_up: CudaTensor::from_cpu(&cpu_ffn.w_up).unwrap(),
            w_down: CudaTensor::from_cpu(&cpu_ffn.w_down).unwrap(),
        };
        let gpu_out = block_inference::<CudaTensor>(&x_gpu, &gpu_heads, &gpu_ffn, head_dim)
            .to_cpu()
            .unwrap();

        assert_eq!(cpu_out.shape, gpu_out.shape, "block output shape mismatch");
        // The block chains rmsnorm -> attention -> add -> rmsnorm ->
        // ffn -> add. That is roughly 12-15 sequential ops; widen the
        // tolerance to 5e-3 to absorb the accumulated drift.
        for (i, (&c, &g)) in cpu_out.data.iter().zip(&gpu_out.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 5e-3 + 5e-3 * c.abs(),
                "block drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    /// **The headline test of Chunk 2.3.** Matches the Chunk 2.2
    /// attention test in spirit: the same generic `ffn_inference<T>`
    /// runs on both backends and the outputs agree within tolerance.
    /// SwiGLU layout: gate_w, up_w map `[hidden, ffn]`; down_w maps
    /// `[ffn, hidden]`. Tolerance is the same `1e-3` chained-op
    /// budget as the attention test.
    #[test]
    fn ffn_inference_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::ffn_inference;
        let seq = 16usize;
        let hidden = 32usize;
        let ffn = 64usize;

        let x = random_tensor(seq, hidden, 2.0);
        let w_gate = random_tensor(hidden, ffn, 2.1);
        let w_up = random_tensor(hidden, ffn, 2.2);
        let w_down = random_tensor(ffn, hidden, 2.3);

        let cpu_out = ffn_inference::<Tensor>(&x, &w_gate, &w_up, &w_down);
        let x_g = CudaTensor::from_cpu(&x).unwrap();
        let g_g = CudaTensor::from_cpu(&w_gate).unwrap();
        let u_g = CudaTensor::from_cpu(&w_up).unwrap();
        let d_g = CudaTensor::from_cpu(&w_down).unwrap();
        let gpu_out = ffn_inference::<CudaTensor>(&x_g, &g_g, &u_g, &d_g)
            .to_cpu()
            .unwrap();

        assert_eq!(cpu_out.shape, gpu_out.shape, "ffn output shape mismatch");
        for (i, (&c, &g)) in cpu_out.data.iter().zip(&gpu_out.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "ffn drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    /// **The headline test of Chunk 2.2.** `attention_head_inference`
    /// is the smallest realistic model layer expressed purely against
    /// the trait surface. Running it once with `Tensor` and once with
    /// `CudaTensor` over the same inputs must produce matching outputs
    /// within the cross-backend tolerance. If any of the six new ops
    /// regresses on either backend, this test catches it.
    #[test]
    fn attention_head_inference_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::attention_head_inference;
        let seq = 16usize;
        let hidden = 32usize;
        let head_dim = 8usize;

        let x = random_tensor(seq, hidden, 1.0);
        let w_q = random_tensor(hidden, head_dim, 1.1);
        let w_k = random_tensor(hidden, head_dim, 1.2);
        let w_v = random_tensor(hidden, head_dim, 1.3);
        let w_o = random_tensor(head_dim, hidden, 1.4);

        let cpu_out = attention_head_inference::<Tensor>(&x, &w_q, &w_k, &w_v, &w_o, head_dim);

        let x_g = CudaTensor::from_cpu(&x).unwrap();
        let q_g = CudaTensor::from_cpu(&w_q).unwrap();
        let k_g = CudaTensor::from_cpu(&w_k).unwrap();
        let v_g = CudaTensor::from_cpu(&w_v).unwrap();
        let o_g = CudaTensor::from_cpu(&w_o).unwrap();
        let gpu_out =
            attention_head_inference::<CudaTensor>(&x_g, &q_g, &k_g, &v_g, &o_g, head_dim)
                .to_cpu()
                .unwrap();

        assert_eq!(
            cpu_out.shape, gpu_out.shape,
            "attention output shape mismatch"
        );
        // Tolerance widened slightly because the chain accumulates
        // drift across 5 matmuls + softmax + RoPE; per-op tolerance is
        // 1e-4 + 1e-4*|val| but the chained product needs ~1e-3.
        for (i, (&c, &g)) in cpu_out.data.iter().zip(&gpu_out.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "attention drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    /// Issue #2: launch overhead dominates the GPU step, so each
    /// quantise call must issue exactly ONE fused kernel launch
    /// (reduction + apply together), not a two-stage pair.
    #[test]
    fn quantise_ste_and_int8_paths_use_one_fused_launch_per_operand() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::{BitLinear, QuantiseActsSTE, QuantiseWeightsSTE};
        let m = 32usize;
        let k = 64usize;
        let n = 16usize;
        let x = CudaTensor::from_cpu(&Tensor::from_vec(
            (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect(),
            vec![m, k],
        ))
        .unwrap();
        let w = CudaTensor::from_cpu(&Tensor::from_vec(
            (0..k * n).map(|i| (i as f32 * 0.09).cos()).collect(),
            vec![k, n],
        ))
        .unwrap();

        // The counter is thread-local, so these deltas are exact even
        // with the rest of the suite hammering the GPU in parallel.
        let before = QUANT_KERNEL_LAUNCHES.with(std::cell::Cell::get);
        let _ = w.quantise_weights_ste();
        let _ = x.quantise_acts_ste();
        let ste_delta = QUANT_KERNEL_LAUNCHES.with(std::cell::Cell::get) - before;
        assert_eq!(
            ste_delta, 2,
            "f32 STE pair must be 1 fused launch per operand, saw {ste_delta}"
        );

        let before = QUANT_KERNEL_LAUNCHES.with(std::cell::Cell::get);
        let _ = x.bit_linear(&w); // k, n multiples of 4: int8 path
        let int8_delta = QUANT_KERNEL_LAUNCHES.with(std::cell::Cell::get) - before;
        assert_eq!(
            int8_delta, 2,
            "int8 bit_linear must quantise with 1 fused launch per operand, saw {int8_delta}"
        );
    }

    /// Phase 5.a quant cross-backend agreement. STE quant is a fixed
    /// transform per row / per tensor, so CPU and GPU outputs should
    /// match to within f32 round-off (the only source of drift is
    /// the order in which the per-row absmax / per-tensor abs-sum
    /// reduction sums its terms; for non-pathological inputs this is
    /// negligible).
    #[test]
    fn quantise_weights_ste_cpu_and_cuda_agree_within_tolerance() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::QuantiseWeightsSTE;
        // Prime dimensions with a wide value range to exercise both
        // round-to-zero and round-to-+/-1 cases.
        let m = 17usize;
        let n = 23usize;
        let w_data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.13).sin() * 1.5 + 0.4)
            .collect();
        let w_cpu = Tensor::from_vec(w_data.clone(), vec![m, n]);
        let w_gpu = CudaTensor::from_cpu(&w_cpu).expect("H->D");
        let q_cpu = w_cpu.quantise_weights_ste();
        let q_gpu = w_gpu.quantise_weights_ste().to_cpu().expect("D->H");
        assert_eq!(q_cpu.shape, q_gpu.shape);
        for (i, (&c, &g)) in q_cpu.data.iter().zip(&q_gpu.data).enumerate() {
            assert!(
                (c - g).abs() <= 1e-5 + 1e-5 * c.abs(),
                "weight quant drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    #[test]
    fn quantise_acts_ste_cpu_and_cuda_agree_within_tolerance() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::QuantiseActsSTE;
        let m = 13usize;
        let n = 19usize;
        let x_data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.071).cos() * 2.0 + 0.3)
            .collect();
        let x_cpu = Tensor::from_vec(x_data.clone(), vec![m, n]);
        let x_gpu = CudaTensor::from_cpu(&x_cpu).expect("H->D");
        let q_cpu = x_cpu.quantise_acts_ste();
        let q_gpu = x_gpu.quantise_acts_ste().to_cpu().expect("D->H");
        assert_eq!(q_cpu.shape, q_gpu.shape);
        for (i, (&c, &g)) in q_cpu.data.iter().zip(&q_gpu.data).enumerate() {
            assert!(
                (c - g).abs() <= 1e-5 + 1e-5 * c.abs(),
                "act quant drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    /// Phase 5.b headline test: the int8 cublasGemmEx BitLinear path
    /// on `CudaTensor` agrees with the CPU `BitLinear` (which is
    /// `quantise_acts_ste(x) @ quantise_weights_ste(w)` via f32
    /// sgemm) within FP tolerance. Both compute the same algebraic
    /// quantity; the int8 path keeps the matmul in int32 (exact for
    /// integer multiplies) and dequantises once at the end, so the
    /// two paths can disagree only at f32 round-off scale.
    #[test]
    fn bit_linear_int8_matches_cpu_within_tolerance() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::BitLinear;
        // v0.13-shape attention Q matmul: [seq, hidden] @ [hidden,
        // head_dim] = [64, 192] @ [192, 16]. Multiples of 16 - the
        // int8 tensor-core path can be used by cuBLAS without
        // dimension padding.
        let m = 64usize;
        let k = 192usize;
        let n = 16usize;
        let x_data: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.041).sin() * 0.5 + 0.1)
            .collect();
        let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.073).cos() * 0.4).collect();
        let x_cpu = Tensor::from_vec(x_data.clone(), vec![m, k]);
        let w_cpu = Tensor::from_vec(w_data.clone(), vec![k, n]);
        let x_gpu = CudaTensor::from_cpu(&x_cpu).expect("H->D x");
        let w_gpu = CudaTensor::from_cpu(&w_cpu).expect("H->D w");
        let y_cpu = x_cpu.bit_linear(&w_cpu);
        let y_gpu = x_gpu.bit_linear(&w_gpu).to_cpu().expect("D->H y");
        assert_eq!(y_cpu.shape, y_gpu.shape);
        // Tolerance: the CPU path runs the matmul in f32 across
        // dequantised values; the GPU path runs it in int32 across
        // raw int8 values then dequantises once. Both produce the
        // same algebraic quantity, but the order in which f32
        // multiplications occur differs, so the last few mantissa
        // bits diverge. Per-cell relative tolerance of ~1% absorbs
        // this without masking real bugs.
        for (i, (&c, &g)) in y_cpu.data.iter().zip(&y_gpu.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 5e-3 + 5e-3 * c.abs(),
                "bit_linear int8 drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    /// **Headline test of Phase 4 chunk 4.5.e.** End-to-end gradient
    /// computation on the GPU. Asserts:
    /// - loss is finite and positive (cross-entropy on random init
    ///   should give ~log(vocab));
    /// - gradient count matches `model.param_shapes().len()`;
    /// - per-gradient shapes match the canonical visitor order;
    /// - every gradient tensor is finite (no NaN / inf);
    /// - the gradient norm is non-zero (a fully-zero set would mean
    ///   the chain dropped a branch somewhere).
    ///
    /// Together with chunks 4.5.a-d's correctness gates, this is
    /// enough sanity that the orchestration is right; tighter
    /// validation lives in chunk 4.5.f's "loss decreases when we
    /// train one step" test.
    #[test]
    fn cuda_model_compute_grads_for_window_smoke() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::model::{Model, ModelConfig};
        let config = ModelConfig {
            vocab_size: 17,
            hidden_dim: 32,
            n_heads: 4,
            head_dim: 8,
            ffn_dim: 64,
            max_seq_len: 16,
            n_blocks: 2,
        };
        let model = Model::new(&config, 42);
        let input: Vec<usize> = vec![0, 3, 7, 11, 5, 1, 9];
        let target: Vec<usize> = vec![3, 7, 11, 5, 1, 9, 14];

        let cuda_model = CudaModel::from_cpu(&model);
        let (grads, loss) = cuda_model.compute_grads_for_window(&input, &target);

        assert!(loss.is_finite(), "loss not finite: {loss}");
        assert!(loss > 0.0, "loss should be positive on random init");
        let shapes = model.param_shapes();
        assert_eq!(
            grads.len(),
            shapes.len(),
            "gradient count mismatch: got {} expected {}",
            grads.len(),
            shapes.len()
        );
        let mut total_norm_sq = 0.0_f64;
        for (i, (g, s)) in grads.iter().zip(&shapes).enumerate() {
            assert_eq!(&g.shape, s, "gradient shape mismatch at param {i}");
            for &v in &g.data {
                assert!(v.is_finite(), "non-finite gradient cell in param {i}");
                total_norm_sq += (v as f64) * (v as f64);
            }
        }
        assert!(
            total_norm_sq > 0.0,
            "all-zero gradient set: chain probably dropped a branch"
        );
    }
}

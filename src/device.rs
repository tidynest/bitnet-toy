//! Device-abstraction traits for CPU + CUDA tensors.
//!
//! Phase 2 keeps activations + per-head weights resident on whichever
//! backend the model is using by writing model code (attention head,
//! later FFN, eventually full forward) generic over a small set of
//! per-op traits. Each trait corresponds to one op family and is
//! implemented for both `Tensor` (CPU) and `CudaTensor` (GPU).
//!
//! Why per-op traits and not a single `Device` trait holding every op:
//! incrementality. Adding a new op (say `Softmax`) is one new trait
//! plus two impls; existing generic code keeps compiling because no
//! existing function signature changes. A single fat `Device` trait
//! would force every backend to implement every op before the trait
//! is satisfied, which we cannot afford with multi-session phased work.
//!
//! Trait methods return `Self` and panic on backend errors. This
//! matches the CPU code's invariant-style asserts (shape mismatch,
//! out-of-bounds, etc. all panic) and means generic helpers do not
//! have to handle a `Result` everywhere. The errors that cuBLAS /
//! cudarc raise (kernel launch failure, allocation failure, lost
//! device, ...) are catastrophic and have no recovery path in this
//! project anyway.
//!
//! Chunk 2.1 (this commit): only `MatMul`. Chunk 2.2 will add
//! `Softmax`, `Add`, `Mul`, `Transpose2D`, `Rope`, `CausalMask` so the
//! full attention head can be written generically.

/// Tiny generic helper used by the Chunk 2.1 cross-backend trait test.
/// Kept around as a minimal smoke test for the trait surface.
#[allow(dead_code)]
pub fn chained_matmul<T: MatMul>(a: &T, b: &T, c: &T) -> T {
    a.matmul(b).matmul(c)
}

/// Per-head attention weight bundle, generic over the backend storage.
/// One bundle per attention head. Owns its tensors (the field types
/// are `T`, not `&T`) so a `Vec<HeadWeights<Tensor>>` and a
/// `Vec<HeadWeights<CudaTensor>>` are independent storage on each
/// backend.
#[allow(dead_code)]
pub struct HeadWeights<T> {
    pub w_q: T,
    pub w_k: T,
    pub w_v: T,
    pub w_o: T,
}

/// SwiGLU FFN weight bundle, generic over the backend storage. Three
/// tensors per block: gate, up (both `[hidden, ffn]`), down (`[ffn,
/// hidden]`).
#[allow(dead_code)]
pub struct FfnWeights<T> {
    pub w_gate: T,
    pub w_up: T,
    pub w_down: T,
}

/// Generic multi-head attention forward, sum-of-projections form.
/// Each head's output (already projected to `[seq, hidden]` by its
/// `w_o`) is summed elementwise into a running total. Mathematically
/// equivalent to the canonical "concat heads then project once with a
/// wide W_o" formulation but does not need a `concat` op.
#[allow(dead_code)]
pub fn multi_head_attention_inference<T>(x: &T, heads: &[HeadWeights<T>], head_dim: usize) -> T
where
    T: MatMul + Add + MulScalar + Transpose2D + Softmax + CausalMask + Rope,
{
    assert!(!heads.is_empty(), "multi_head_attention requires at least one head");
    let mut combined = attention_head_inference(
        x, &heads[0].w_q, &heads[0].w_k, &heads[0].w_v, &heads[0].w_o, head_dim,
    );
    for h in &heads[1..] {
        let head_out =
            attention_head_inference(x, &h.w_q, &h.w_k, &h.w_v, &h.w_o, head_dim);
        combined = combined.add(&head_out);
    }
    combined
}

/// Generic transformer block forward (pre-norm, LLaMA / BitNet style):
///     x1 = x  + attention(rmsnorm(x))
///     x2 = x1 + ffn(rmsnorm(x1))
///
/// Compiles for any `T` that implements the full trait set. The
/// inference-only path (no quant, no autograd) - this is the chunk-2.4
/// surface that Phase 3 wires into a `CudaModel` for end-to-end
/// device-resident forward / sampling.
#[allow(dead_code)]
pub fn block_inference<T>(
    x: &T,
    heads: &[HeadWeights<T>],
    ffn: &FfnWeights<T>,
    head_dim: usize,
) -> T
where
    T: MatMul
        + Add
        + MulScalar
        + Transpose2D
        + Softmax
        + CausalMask
        + Rope
        + RmsNorm
        + Silu
        + Mul,
{
    let attn_out = multi_head_attention_inference(&x.rmsnorm(), heads, head_dim);
    let x1 = x.add(&attn_out);
    let ffn_out = ffn_inference(&x1.rmsnorm(), &ffn.w_gate, &ffn.w_up, &ffn.w_down);
    x1.add(&ffn_out)
}

/// Generic SwiGLU FFN forward pass, written against the trait surface.
/// Same math as the autograd `ffn::ffn` (LLaMA / BitNet b1.58 form):
///
///     gate = silu(x @ W_gate)
///     up   = x @ W_up
///     h    = gate * up
///     y    = h @ W_down
///
/// Compiles for any backend `T` that implements the trait set. No
/// BitNet quantisation in this Phase 2 path; Phase 4 adds GPU autograd
/// + quant kernels and wires this into training.
#[allow(dead_code)]
pub fn ffn_inference<T>(x: &T, w_gate: &T, w_up: &T, w_down: &T) -> T
where
    T: MatMul + Silu + Mul,
{
    let gate = x.matmul(w_gate).silu();
    let up = x.matmul(w_up);
    gate.mul(&up).matmul(w_down)
}

/// Generic forward pass for one attention head, written purely against
/// the trait surface declared below. Compiles for any backend `T` that
/// implements the full op set; `Tensor` and `CudaTensor` both do.
///
/// Math (per the BitNet b1.58 / LLaMA convention):
///     Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
///     Q = rope(Q),  K = rope(K)
///     scores = (Q @ K.T) * (1 / sqrt(head_dim))
///     scores = causal_mask(scores)
///     attn   = softmax(scores)
///     ctx    = attn @ V
///     out    = ctx @ W_o
///
/// Inference path only: no BitNet quantisation, no autograd. Phase 2.2
/// proves the trait architecture scales to a real model layer; the
/// production training path stays on the existing `Var`-based
/// `attention::attention` until Phase 4 adds GPU autograd.
#[allow(dead_code)]
pub fn attention_head_inference<T>(
    x: &T,
    w_q: &T,
    w_k: &T,
    w_v: &T,
    w_o: &T,
    head_dim: usize,
) -> T
where
    T: MatMul + MulScalar + Transpose2D + Softmax + CausalMask + Rope,
{
    let q = x.matmul(w_q).rope();
    let k = x.matmul(w_k).rope();
    let v = x.matmul(w_v);

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let scores = q.matmul(&k.transpose_2d()).mul_scalar(scale).causal_mask();
    let attn = scores.softmax();
    let ctx = attn.matmul(&v);
    ctx.matmul(w_o)
}

/// Row-major 2-D matrix multiplication: `out = self @ rhs`.
///
/// Implementors must accept `self.shape[1] == rhs.shape[0]` and
/// produce output of shape `[self.shape[0], rhs.shape[1]]`. Shape
/// mismatches panic. The backend is free to use any internal
/// summation order, so the bit-pattern of every output cell is
/// implementation-defined; the only cross-backend guarantee is that
/// the result is mathematically the same matrix product up to f32
/// round-off (~`1e-4 + 1e-4 * |val|` empirical tolerance between
/// CPU and CUDA paths).
pub trait MatMul {
    /// Mathematical matrix product. Returns a freshly-owned tensor on
    /// the same backend as `self`.
    fn matmul(&self, rhs: &Self) -> Self;
}

/// Elementwise add: `out[i] = self[i] + rhs[i]`. Shapes must match.
/// Used by FFN residual connections (block-level) - not by attention,
/// which is why it currently looks unused in non-test builds. Tests
/// exercise both backends.
#[allow(dead_code)]
pub trait Add {
    fn add(&self, rhs: &Self) -> Self;
}

/// Multiply every cell by a `f32` scalar: `out[i] = self[i] * s`.
pub trait MulScalar {
    fn mul_scalar(&self, s: f32) -> Self;
}

/// 2-D transpose: `[r, c] -> [c, r]`. Physically copies because tensors
/// here are dense + strideless.
pub trait Transpose2D {
    fn transpose_2d(&self) -> Self;
}

/// Per-row softmax over the last axis (input shape `[m, n]`, output
/// `[m, n]`). Uses the standard max-subtraction trick for numerical
/// stability so an outlier logit cannot blow up `exp`.
pub trait Softmax {
    fn softmax(&self) -> Self;
}

/// Causal (upper-triangular) attention mask: `out[i, j] = -inf` for
/// `j > i`, else `out[i, j] = self[i, j]`. Input shape `[seq, seq]`.
/// Applied to scores BEFORE softmax so that `softmax(-inf) = 0` keeps
/// queries from attending to future keys.
pub trait CausalMask {
    fn causal_mask(&self) -> Self;
}

/// Rotary Position Embedding. Input shape `[seq, head_dim]`; head_dim
/// must be even. Each row `pos in 0..seq` and pair index `i in
/// 0..head_dim/2` gets the 2-D vector `(x[pos, 2i], x[pos, 2i+1])`
/// rotated by angle `pos * 10000^(-2i / head_dim)`. Parameter-free.
pub trait Rope {
    fn rope(&self) -> Self;
}

/// Sigmoid Linear Unit activation: `silu(x) = x / (1 + exp(-x))`.
/// Smooth, differentiable everywhere; the activation function used by
/// SwiGLU's gate branch.
pub trait Silu {
    fn silu(&self) -> Self;
}

/// Elementwise multiply: `out[i] = self[i] * rhs[i]`. Shapes must match.
/// Used by SwiGLU's gate * up step inside the FFN.
pub trait Mul {
    fn mul(&self, rhs: &Self) -> Self;
}

/// Per-row RMS normalisation (no learnable gain). Input `[m, n]`,
/// output same shape. For each row:
///     rms = sqrt(mean_j(x[i, j]^2) + EPS)     # EPS = 1e-5
///     y[i, j] = x[i, j] / rms
/// LLaMA / BitNet b1.58 convention. Pre-normalises every sublayer
/// input inside a transformer block.
pub trait RmsNorm {
    fn rmsnorm(&self) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Generic helper compiles + runs on `Tensor`. Establishes the
    /// pattern; does not rely on any backend-specific behaviour.
    #[test]
    fn chained_matmul_works_on_cpu_tensor() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let result = chained_matmul(&a, &b, &c);
        // a @ b @ I should equal a @ b. Manual: a@b = [[19,22],[43,50]].
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    /// **The point of Chunk 2.1.** A single function written against
    /// the `MatMul` trait runs on both backends and the outputs agree
    /// within FP tolerance. If the trait surface ever drifts (or one
    /// impl regresses) this test catches it before any model code
    /// builds on the abstraction.
    #[cfg(feature = "cuda")]
    #[test]
    fn chained_matmul_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 32usize;
        let k = 24usize;
        let n = 16usize;
        let p = 8usize;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.071).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.103).cos()).collect();
        let c_data: Vec<f32> = (0..n * p).map(|i| (i as f32 * 0.037).sin()).collect();

        let a_cpu = Tensor::from_vec(a_data.clone(), vec![m, k]);
        let b_cpu = Tensor::from_vec(b_data.clone(), vec![k, n]);
        let c_cpu = Tensor::from_vec(c_data.clone(), vec![n, p]);
        let cpu_out = chained_matmul(&a_cpu, &b_cpu, &c_cpu);

        let a_gpu = CudaTensor::from_cpu(&a_cpu).expect("H->D failed");
        let b_gpu = CudaTensor::from_cpu(&b_cpu).expect("H->D failed");
        let c_gpu = CudaTensor::from_cpu(&c_cpu).expect("H->D failed");
        let gpu_out = chained_matmul(&a_gpu, &b_gpu, &c_gpu)
            .to_cpu()
            .expect("D->H failed");

        assert_eq!(cpu_out.shape, gpu_out.shape, "shape diverged");
        let tol_abs: f32 = 1e-4;
        let tol_rel: f32 = 1e-4;
        for (i, (&c, &g)) in cpu_out.data.iter().zip(&gpu_out.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= tol_abs + tol_rel * c.abs(),
                "trait helper drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }
}

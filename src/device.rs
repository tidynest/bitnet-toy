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

/// Generic matmul backward: given `c = a @ b` and incoming gradient `grad_c`,
/// produce `(grad_a, grad_b)` via the closed-form identity that matches the
/// CPU-side `Var::matmul` (autograd.rs:253-260):
///
///     grad_a = grad_c @ b.T          shapes [m, n] @ [n, k] -> [m, k]
///     grad_b = a.T @ grad_c          shapes [k, m] @ [m, n] -> [k, n]
///
/// Pure trait composition: zero new kernels are needed because the building
/// blocks (`MatMul`, `Transpose2D`) already work on both backends. This is
/// the smallest credible Phase 4 chunk - it lets backward propagation pass
/// through any model layer that currently chains matmuls without adding any
/// CUDA-side autograd machinery.
#[allow(dead_code)]
pub fn matmul_backward<T>(grad_c: &T, a: &T, b: &T) -> (T, T)
where
    T: MatMul + Transpose2D,
{
    let grad_a = grad_c.matmul(&b.transpose_2d());
    let grad_b = a.transpose_2d().matmul(grad_c);
    (grad_a, grad_b)
}

/// Elementwise add backward: given `c = a + b` and incoming gradient
/// `grad_c`, both inputs receive `grad_c` unchanged. Returns two
/// independent owned clones so each can be accumulated into a separate
/// upstream gradient buffer without aliasing. Trait bound is `Clone`
/// only (no new kernels). For `CudaTensor` this triggers a
/// device-to-device memcpy via the `Clone` impl on `cuda.rs`.
#[allow(dead_code)]
pub fn add_backward<T: Clone>(grad_c: &T) -> (T, T) {
    (grad_c.clone(), grad_c.clone())
}

/// Elementwise multiply backward: given `c = a * b` (Hadamard product)
/// and incoming gradient `grad_c`, the per-input gradients are
///
///     grad_a = grad_c * b
///     grad_b = grad_c * a
///
/// Pure composition over the existing `Mul` trait; no new kernel needed.
/// Each input is read twice but the output is freshly allocated, so the
/// two `Mul` calls do not alias. Order: a-then-b (mirrors `matmul_backward`).
#[allow(dead_code)]
pub fn mul_backward<T: Mul>(grad_c: &T, a: &T, b: &T) -> (T, T) {
    let grad_a = grad_c.mul(b);
    let grad_b = grad_c.mul(a);
    (grad_a, grad_b)
}

/// Scalar-multiply backward: given `c = a * s` (broadcast f32 scalar)
/// and incoming gradient `grad_c`, the upstream tensor gradient is
///
///     grad_a = grad_c * s
///
/// The scalar `s` is not a tape leaf in this project (it is a pure
/// `f32` knob, never differentiated against), so there is no second
/// return value. Pure composition over `MulScalar`; no new kernel.
#[allow(dead_code)]
pub fn mul_scalar_backward<T: MulScalar>(grad_c: &T, s: f32) -> T {
    grad_c.mul_scalar(s)
}

/// SiLU backward: per-cell activation gradient
///
///     d(silu)/dx = sig * (1 + x * (1 - sig))            where sig = sigmoid(x)
///     grad_x[i]  = grad_y[i] * d(silu)/dx [at x[i]]
///
/// Matches the CPU `Var::silu` backward (autograd.rs:579-590) exactly.
/// One fused kernel is much cheaper than the four-or-five-launch path
/// that decomposing into existing ops (mul, mul_scalar, sub, add, plus a
/// missing `Sigmoid`) would require, so this gets its own trait + kernel.
pub trait SiluBackward {
    /// `self` is the upstream gradient `grad_y`; `x` is the saved forward
    /// input. Returns `grad_x` with the same shape.
    fn silu_backward(&self, x: &Self) -> Self;
}

/// Generic SiLU backward helper. Argument order mirrors `Var::silu`'s
/// closure: incoming gradient first, saved input second; returns the
/// upstream gradient.
#[allow(dead_code)]
pub fn silu_backward<T: SiluBackward>(grad_y: &T, x: &T) -> T {
    grad_y.silu_backward(x)
}

/// Per-row softmax backward (Phase 4 chunk 4.3). The softmax Jacobian is
/// `J = diag(s) - s s^T` per row, so the JVP collapses to
///
///     grad_in[i, j] = s[i, j] * (grad_y[i, j] - sum_k grad_y[i, k] * s[i, k])
///
/// Crucially, the formula uses the saved **output** `s` (not the input
/// `x`) - the autograd `Var::softmax` saves output for exactly this
/// reason (autograd.rs:618-621). Each row needs one reduction (`dot`)
/// before the cell-wise pass, so this gets its own fused kernel rather
/// than decomposing.
pub trait SoftmaxBackward {
    /// `self` is the upstream gradient `grad_y`; `s_out` is the saved
    /// softmax forward output. Both shape `[m, n]`. Returns `grad_x`
    /// with the same shape.
    fn softmax_backward(&self, s_out: &Self) -> Self;
}

/// Causal-mask backward (Phase 4 chunk 4.3). Forward sets the upper
/// triangle (`j > i`) to `-inf`, which after softmax becomes zero
/// probability. Backward zeros gradient at those positions and passes
/// the lower triangle through unchanged - matches `Var::causal_mask`
/// (autograd.rs:778-819). No saved tensor: the mask pattern is fixed
/// by shape.
pub trait CausalMaskBackward {
    /// `self` is the upstream gradient `grad_y`; shape `[seq, seq]`.
    /// Returns `grad_x` with `grad_x[i, j] = grad_y[i, j]` for
    /// `j <= i`, else `0`.
    fn causal_mask_backward(&self) -> Self;
}

/// Generic softmax backward helper. Argument order mirrors the autograd
/// closure: upstream gradient first, saved softmax output second;
/// returns the upstream gradient flowing into the softmax input.
#[allow(dead_code)]
pub fn softmax_backward<T: SoftmaxBackward>(grad_y: &T, s_out: &T) -> T {
    grad_y.softmax_backward(s_out)
}

/// Generic causal-mask backward helper. The forward output is not
/// saved (the mask is shape-determined), so the helper takes only the
/// upstream gradient.
#[allow(dead_code)]
pub fn causal_mask_backward<T: CausalMaskBackward>(grad_y: &T) -> T {
    grad_y.causal_mask_backward()
}

/// Per-row RMSNorm backward (Phase 4 chunk 4.4). Forward is
/// `y[i, j] = x[i, j] / rms_i` with `rms_i = sqrt(mean_j(x[i, j]^2) +
/// EPS)`. Backward derives via quotient + chain rule:
///
///     inv_rms_i        = 1 / rms_i
///     dot_i            = sum_j x_saved[i, j] * grad_y[i, j]
///     factor_i         = dot_i * inv_rms_i^3 / n
///     grad_in[i, j]    = grad_y[i, j] * inv_rms_i - x_saved[i, j] * factor_i
///
/// Matches the CPU `Var::rmsnorm` closure (autograd.rs:702-758)
/// cell-for-cell. Recomputing `rms` from `x_saved` inside the kernel
/// is cheaper than carrying a separate `[m]` tensor of saved row
/// norms across the trait surface, and keeps the API symmetric with
/// `softmax_backward(grad_y, s_out)`.
pub trait RmsNormBackward {
    /// `self` is the upstream gradient `grad_y`; `x_saved` is the
    /// saved forward input. Both shape `[m, n]`. Returns `grad_x`.
    fn rmsnorm_backward(&self, x_saved: &Self) -> Self;
}

/// RoPE backward (Phase 4 chunk 4.4). Forward rotates each pair
/// `(x[pos, 2i], x[pos, 2i+1])` by `pos * 10000^(-2i/head_dim)`;
/// backward applies the inverse rotation (same trig table with `sin`
/// flipped, since each per-pair rotation is orthogonal):
///
///     grad_in[pos, 2i]   =  grad_y[pos, 2i]   * cos + grad_y[pos, 2i+1] * sin
///     grad_in[pos, 2i+1] = -grad_y[pos, 2i]   * sin + grad_y[pos, 2i+1] * cos
///
/// No saved tensor: the angles are shape-determined from
/// `[seq, head_dim]`. Matches the CPU `Var::rope` closure
/// (autograd.rs:902-925).
pub trait RopeBackward {
    /// `self` is the upstream gradient `grad_y`; shape `[seq, head_dim]`
    /// with `head_dim` even. Returns `grad_x` of the same shape.
    fn rope_backward(&self) -> Self;
}

/// Generic RMSNorm backward helper. Argument order mirrors the
/// autograd closure: upstream gradient first, saved forward input
/// second; returns the upstream gradient flowing into the rmsnorm
/// input.
#[allow(dead_code)]
pub fn rmsnorm_backward<T: RmsNormBackward>(grad_y: &T, x_saved: &T) -> T {
    grad_y.rmsnorm_backward(x_saved)
}

/// Generic RoPE backward helper. Single argument because the rotation
/// angles depend only on shape, not on the saved forward input.
#[allow(dead_code)]
pub fn rope_backward<T: RopeBackward>(grad_y: &T) -> T {
    grad_y.rope_backward()
}

/// Cross-entropy loss + softmax forward (Phase 4 chunk 4.5.d). Closed-
/// form mean cross-entropy over a `[seq, vocab]` logits tensor with a
/// `[seq]` target-index slice. Returns `(loss_scalar, softmax_output)`
/// in one fused pass: the forward computes softmax for every row, and
/// the loss `mean_i(-log(softmax[i, target[i]]))` falls out.
///
/// Why fuse: the backward needs the saved softmax output, and computing
/// softmax again from logits would mean a second pass through the same
/// expensive `exp` (and the same log-sum-exp numerical-stability dance).
///
/// Matches the CPU `Var::cross_entropy` (autograd.rs:1025-1115) byte-
/// for-byte (subtract-max log-sum-exp trick, mean-over-seq normalisation).
pub trait CrossEntropy {
    /// `self` is the logits `[seq, vocab]`; `targets` is a `[seq]` slice
    /// of class indices in `0..vocab`. Returns the scalar loss and the
    /// per-row softmax output (shape `[seq, vocab]`).
    fn cross_entropy_forward_save(&self, targets: &[usize]) -> (f32, Self)
    where
        Self: Sized;
}

/// Cross-entropy gradient flowing back into the logits (Phase 4 chunk
/// 4.5.d). Closed form per cell:
///
///     grad_logits[i, j] = (softmax_saved[i, j] - delta_{j, target[i]}) / seq
///
/// `self` is the saved softmax output from `cross_entropy_forward_save`.
/// The default `Tape::backward` seed of `g_scalar = 1` is implicitly
/// folded in (the helper assumes the loss has already been turned into
/// a scalar via the mean-over-seq inside the forward).
pub trait CrossEntropyBackward {
    fn cross_entropy_backward(&self, targets: &[usize], seq: usize) -> Self;
}

/// Generic helpers for cross-entropy forward + backward. The trait
/// methods take their canonical arguments; the free-function helpers
/// follow the chunk-4.x naming convention so call sites read like the
/// rest of Phase 4.
#[allow(dead_code)]
pub fn cross_entropy_forward_save<T: CrossEntropy>(logits: &T, targets: &[usize]) -> (f32, T) {
    logits.cross_entropy_forward_save(targets)
}

#[allow(dead_code)]
pub fn cross_entropy_backward<T: CrossEntropyBackward>(
    softmax_saved: &T,
    targets: &[usize],
    seq: usize,
) -> T {
    softmax_saved.cross_entropy_backward(targets, seq)
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
    assert!(
        !heads.is_empty(),
        "multi_head_attention requires at least one head"
    );
    let mut combined = attention_head_inference(
        x,
        &heads[0].w_q,
        &heads[0].w_k,
        &heads[0].w_v,
        &heads[0].w_o,
        head_dim,
    );
    for h in &heads[1..] {
        let head_out = attention_head_inference(x, &h.w_q, &h.w_k, &h.w_v, &h.w_o, head_dim);
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
    T: MatMul + Add + MulScalar + Transpose2D + Softmax + CausalMask + Rope + RmsNorm + Silu + Mul,
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

/// Phase 4 chunk 4.5.b: forward intermediates the SwiGLU FFN backward
/// needs to retain. Selection rule (same as chunk 4.5.a):
/// - `gate_pre` is the input to silu, needed for `silu_backward`.
/// - `gate` is the silu output, needed for `mul_backward(grad_h,
///   gate, up)`.
/// - `up` is needed for `mul_backward(grad_h, gate, up)`.
/// - `h = gate * up` is needed for `matmul_backward(grad_y, h,
///   w_down)`.
///
/// All four are owned f32 tensors on whichever backend ran the
/// forward; the CPU `Clone` derive cost is trivial, the CUDA `Clone`
/// path is never invoked because forward `move`s the intermediates
/// into the saved struct rather than copying them.
#[allow(dead_code)]
pub struct FfnSaved<T> {
    pub gate_pre: T,
    pub gate: T,
    pub up: T,
    pub h: T,
}

/// Phase 4 chunk 4.5.b: gradients the SwiGLU FFN produces. `grad_x`
/// is the residual sum of two branches (`x` feeds both `w_gate` and
/// `w_up`); the three `grad_w_*` tensors are the per-weight gradients
/// in canonical visitor order (gate / up / down).
#[allow(dead_code)]
pub struct FfnGrads<T> {
    pub grad_x: T,
    pub grad_w_gate: T,
    pub grad_w_up: T,
    pub grad_w_down: T,
}

/// Forward pass through the SwiGLU FFN, returning every intermediate
/// the backward needs. Identical math to `ffn_inference`; the only
/// difference is owning the intermediates in the returned saved
/// struct rather than dropping them at end-of-expression.
#[allow(dead_code)]
pub fn ffn_forward_save<T>(x: &T, w_gate: &T, w_up: &T, w_down: &T) -> (T, FfnSaved<T>)
where
    T: MatMul + Silu + Mul,
{
    let gate_pre = x.matmul(w_gate);
    let gate = gate_pre.silu();
    let up = x.matmul(w_up);
    let h = gate.mul(&up);
    let out = h.matmul(w_down);
    (
        out,
        FfnSaved {
            gate_pre,
            gate,
            up,
            h,
        },
    )
}

/// Backward pass through the SwiGLU FFN, hand-traced from the forward
/// in `ffn_forward_save`. Walks the autograd graph in reverse-creation
/// order:
///
///     y = h @ w_down                                     (matmul_backward)
///     h = gate * up                                      (mul_backward)
///     gate = silu(gate_pre)                              (silu_backward)
///     up = x @ w_up;  gate_pre = x @ w_gate              (two matmul_backwards)
///     grad_x = grad_x_gate + grad_x_up                   (residual sum)
///
/// Trait bounds grow over `ffn_forward_save`'s set by `Add` (for the
/// residual `grad_x` sum), `Transpose2D` (used by every
/// `matmul_backward` call), and `SiluBackward` (chunk 4.2). `Mul` is
/// already there from forward; `MatMul` likewise.
#[allow(dead_code)]
pub fn ffn_backward<T>(
    grad_y: &T,
    x: &T,
    saved: &FfnSaved<T>,
    w_gate: &T,
    w_up: &T,
    w_down: &T,
) -> FfnGrads<T>
where
    T: MatMul + Mul + Add + Transpose2D + SiluBackward,
{
    // y = h @ w_down
    let (grad_h, grad_w_down) = matmul_backward(grad_y, &saved.h, w_down);
    // h = gate * up
    let (grad_gate, grad_up) = mul_backward(&grad_h, &saved.gate, &saved.up);
    // gate = silu(gate_pre)
    let grad_gate_pre = silu_backward(&grad_gate, &saved.gate_pre);
    // up = x @ w_up
    let (grad_x_up, grad_w_up) = matmul_backward(&grad_up, x, w_up);
    // gate_pre = x @ w_gate
    let (grad_x_gate, grad_w_gate) = matmul_backward(&grad_gate_pre, x, w_gate);
    // x feeds both branches; gradients sum.
    let grad_x = grad_x_gate.add(&grad_x_up);
    FfnGrads {
        grad_x,
        grad_w_gate,
        grad_w_up,
        grad_w_down,
    }
}

/// Phase 4 chunk 4.5.c: per-head weight gradients only. The block-level
/// `BlockGrads<T>` stores one of these per head; the per-head input
/// gradient is *not* stored individually because each head's input
/// gradient is added into the shared `grad_y1_pre` accumulator inside
/// `block_backward`. Keeping the struct narrow makes the block
/// gradient bundle cleaner to consume from a training loop.
#[allow(dead_code)]
pub struct HeadWeightGrads<T> {
    pub grad_w_q: T,
    pub grad_w_k: T,
    pub grad_w_v: T,
    pub grad_w_o: T,
}

/// Phase 4 chunk 4.5.c: forward intermediates the block backward needs.
/// Holds:
/// - `y1_pre = rmsnorm(x)`: input to MHA, needed for
///   `attention_head_backward` + the upstream `rmsnorm_backward`.
/// - `x1 = x + y1`: residual output, needed as the saved input for
///   the second `rmsnorm_backward`.
/// - `y2_pre = rmsnorm(x1)`: input to FFN, needed for `ffn_backward`.
/// - `head_saveds`: per-head Saved structs from chunk 4.5.a.
/// - `ffn_saved`: FFN Saved struct from chunk 4.5.b.
///
/// Note: the block input `x` itself is not stored - it is passed
/// separately into `block_backward` (matching the convention used by
/// `attention_head_backward` and `ffn_backward`).
#[allow(dead_code)]
pub struct BlockSaved<T> {
    pub y1_pre: T,
    pub x1: T,
    pub y2_pre: T,
    pub head_saveds: Vec<AttentionHeadSaved<T>>,
    pub ffn_saved: FfnSaved<T>,
}

/// Phase 4 chunk 4.5.c: gradients one transformer block produces.
/// `grad_x` is the residual sum of two branches (the input feeds both
/// the attention residual `x1 = x + y1` and the rmsnorm before MHA),
/// `head_grads` is one entry per attention head in canonical order,
/// and the three `grad_w_*` fields cover the FFN weights.
#[allow(dead_code)]
pub struct BlockGrads<T> {
    pub grad_x: T,
    pub head_grads: Vec<HeadWeightGrads<T>>,
    pub grad_w_gate: T,
    pub grad_w_up: T,
    pub grad_w_down: T,
}

/// Forward pass through one transformer block, returning every
/// intermediate the backward needs. Identical math to `block_inference`
/// (chunk 2.4); the only difference is owning the intermediates in
/// the returned `BlockSaved<T>` rather than dropping them at end-of-
/// expression. Pre-norm residual structure:
///
///     y1_pre = rmsnorm(x);   y1 = MHA(y1_pre)
///     x1     = x + y1
///     y2_pre = rmsnorm(x1);  y2 = ffn(y2_pre)
///     out    = x1 + y2
#[allow(dead_code)]
pub fn block_forward_save<T>(
    x: &T,
    heads: &[HeadWeights<T>],
    ffn: &FfnWeights<T>,
    head_dim: usize,
) -> (T, BlockSaved<T>)
where
    T: MatMul + Add + MulScalar + Transpose2D + Softmax + CausalMask + Rope + RmsNorm + Silu + Mul,
{
    assert!(!heads.is_empty(), "block_forward_save: at least one head");
    let y1_pre = x.rmsnorm();
    // Per-head forward; collect saved structs and accumulate sum-of-
    // projections into y1.
    let mut head_saveds: Vec<AttentionHeadSaved<T>> = Vec::with_capacity(heads.len());
    let (head0_out, head0_saved) = attention_head_forward_save(
        &y1_pre,
        &heads[0].w_q,
        &heads[0].w_k,
        &heads[0].w_v,
        &heads[0].w_o,
        head_dim,
    );
    head_saveds.push(head0_saved);
    let mut y1 = head0_out;
    for h in &heads[1..] {
        let (out_h, saved_h) =
            attention_head_forward_save(&y1_pre, &h.w_q, &h.w_k, &h.w_v, &h.w_o, head_dim);
        y1 = y1.add(&out_h);
        head_saveds.push(saved_h);
    }
    let x1 = x.add(&y1);
    let y2_pre = x1.rmsnorm();
    let (y2, ffn_saved) = ffn_forward_save(&y2_pre, &ffn.w_gate, &ffn.w_up, &ffn.w_down);
    let out = x1.add(&y2);
    let saved = BlockSaved {
        y1_pre,
        x1,
        y2_pre,
        head_saveds,
        ffn_saved,
    };
    (out, saved)
}

/// Backward pass through one transformer block, hand-traced from the
/// forward in `block_forward_save`. The two residual `x + y` adds
/// give identity gradient flow on both branches via `add_backward`
/// (Clone), so the block backward composes naturally:
///
///     out    = x1 + y2                    (add_backward)
///     y2     = ffn(y2_pre, ...)           (ffn_backward)
///     y2_pre = rmsnorm(x1)                (rmsnorm_backward, saves x1)
///     x1     = x + y1                     (add_backward)
///     y1     = sum_h MHA_h(y1_pre)        (per-head attention_head_backward;
///                                          each head receives the SAME grad_y1
///                                          because `y1` is a sum, and head
///                                          gradients to `y1_pre` accumulate)
///     y1_pre = rmsnorm(x)                 (rmsnorm_backward, saves x)
///
/// `grad_x` ends up as the sum of two branches: one from
/// `x1 = x + y1`, one from `y1_pre = rmsnorm(x)`.
#[allow(dead_code)]
pub fn block_backward<T>(
    grad_out: &T,
    x: &T,
    saved: &BlockSaved<T>,
    heads: &[HeadWeights<T>],
    ffn: &FfnWeights<T>,
    head_dim: usize,
) -> BlockGrads<T>
where
    T: MatMul
        + Add
        + Mul
        + MulScalar
        + Transpose2D
        + SoftmaxBackward
        + CausalMaskBackward
        + RopeBackward
        + RmsNormBackward
        + SiluBackward
        + Clone,
{
    assert!(!heads.is_empty(), "block_backward: at least one head");
    assert_eq!(
        heads.len(),
        saved.head_saveds.len(),
        "block_backward: head count mismatch (weights {} vs saved {})",
        heads.len(),
        saved.head_saveds.len()
    );

    // out = x1 + y2
    let (grad_x1_a, grad_y2) = add_backward(grad_out);
    // y2 = ffn(y2_pre, ...)
    let ffn_grads = ffn_backward(
        &grad_y2,
        &saved.y2_pre,
        &saved.ffn_saved,
        &ffn.w_gate,
        &ffn.w_up,
        &ffn.w_down,
    );
    // y2_pre = rmsnorm(x1)
    let grad_x1_b = rmsnorm_backward(&ffn_grads.grad_x, &saved.x1);
    let grad_x1 = grad_x1_a.add(&grad_x1_b);

    // x1 = x + y1
    let (grad_x_a, grad_y1) = add_backward(&grad_x1);
    // y1 = sum_h MHA_h(y1_pre). Each head's output gets the same
    // grad_y1; per-head x-gradients accumulate into a shared
    // grad_y1_pre.
    let mut head_grads: Vec<HeadWeightGrads<T>> = Vec::with_capacity(heads.len());
    let mut grad_y1_pre: Option<T> = None;
    for (saved_h, h) in saved.head_saveds.iter().zip(heads.iter()) {
        let g = attention_head_backward(
            &grad_y1,
            &saved.y1_pre,
            saved_h,
            &h.w_q,
            &h.w_k,
            &h.w_v,
            &h.w_o,
            head_dim,
        );
        grad_y1_pre = Some(match grad_y1_pre {
            None => g.grad_x,
            Some(acc) => acc.add(&g.grad_x),
        });
        head_grads.push(HeadWeightGrads {
            grad_w_q: g.grad_w_q,
            grad_w_k: g.grad_w_k,
            grad_w_v: g.grad_w_v,
            grad_w_o: g.grad_w_o,
        });
    }
    let grad_y1_pre = grad_y1_pre.expect("block_backward: at least one head");
    // y1_pre = rmsnorm(x)
    let grad_x_b = rmsnorm_backward(&grad_y1_pre, x);
    let grad_x = grad_x_a.add(&grad_x_b);

    BlockGrads {
        grad_x,
        head_grads,
        grad_w_gate: ffn_grads.grad_w_gate,
        grad_w_up: ffn_grads.grad_w_up,
        grad_w_down: ffn_grads.grad_w_down,
    }
}

// ---- Phase 5.a: bitnet variants of the chunk-4.5.x layer helpers ----
//
// Same forward / backward shape as the chunk-4.5.x helpers, but the
// "matmul of acts against learnable weights" pattern is replaced with
// `bit_linear` (quantise_acts_ste(x) @ quantise_weights_ste(w)). The
// internal "matmul of activations against activations" matmuls (q@k.T,
// attn@v) stay unquantised - matches the autograd attention.rs and
// ffn.rs paths exactly. Saved struct shapes are identical to the
// chunk-4.5.x versions because the saved tensors are post-matmul (or
// post-rope / post-softmax) values; the quantisation happens INSIDE
// bit_linear and is invisible to the saved struct.
//
// Backward calls `bit_linear_backward` instead of `matmul_backward` at
// the four / three call sites where bit_linear was used in forward;
// other backward calls (transpose, softmax, causal_mask, rope) are
// unchanged.

/// Phase 5.a: BitNet attention head forward+save. Same math as
/// `attention_head_forward_save` (chunk 4.5.a) but with STE quant
/// inserted before each matmul of activations against learnable
/// weights (W_q / W_k / W_v / W_o). The pre-softmax scores matmul
/// (q @ k.T) and the context matmul (attn @ v) stay unquantised -
/// they are activation-against-activation, not against learnable
/// weights. Matches the autograd `attention::head_output` path.
#[allow(dead_code)]
pub fn attention_head_forward_save_bitnet<T>(
    x: &T,
    w_q: &T,
    w_k: &T,
    w_v: &T,
    w_o: &T,
    head_dim: usize,
) -> (T, AttentionHeadSaved<T>)
where
    T: MatMul + MulScalar + Transpose2D + Softmax + CausalMask + Rope + BitLinear,
{
    let q = x.bit_linear(w_q).rope();
    let k = x.bit_linear(w_k).rope();
    let v = x.bit_linear(w_v);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let scores = q.matmul(&k.transpose_2d()).mul_scalar(scale).causal_mask();
    let attn = scores.softmax();
    let ctx = attn.matmul(&v);
    let out = ctx.bit_linear(w_o);
    let saved = AttentionHeadSaved { q, k, v, attn, ctx };
    (out, saved)
}

/// Phase 5.a: BitNet attention head backward. Same chain as
/// `attention_head_backward` (chunk 4.5.a) but the four matmul_backward
/// calls that correspond to bit_linear forwards are replaced with
/// `bit_linear_backward`; the act-against-act matmuls keep
/// `matmul_backward`. STE makes both quants identity in backward, so
/// the gradient flowing into the original (pre-quant) `x` and `w_*`
/// tensors is the same as if quant didn't exist - we just need to
/// reconstruct the quantised values during backward to feed
/// `matmul_backward`'s `dA = grad_y @ B.T`, `dB = A.T @ grad_y` formula.
#[allow(dead_code)]
pub fn attention_head_backward_bitnet<T>(
    grad_out: &T,
    x: &T,
    saved: &AttentionHeadSaved<T>,
    w_q: &T,
    w_k: &T,
    w_v: &T,
    w_o: &T,
    head_dim: usize,
) -> AttentionHeadGrads<T>
where
    T: MatMul
        + Add
        + MulScalar
        + Transpose2D
        + SoftmaxBackward
        + CausalMaskBackward
        + RopeBackward
        + QuantiseActsSTE
        + QuantiseWeightsSTE,
{
    // out = bit_linear(ctx, w_o)
    let (grad_ctx, grad_w_o) = bit_linear_backward(grad_out, &saved.ctx, w_o);
    // ctx = attn @ v   (act @ act, NOT quantised)
    let (grad_attn, grad_v) = matmul_backward(&grad_ctx, &saved.attn, &saved.v);
    let grad_scores_msk = softmax_backward(&grad_attn, &saved.attn);
    let grad_scores_scl = causal_mask_backward(&grad_scores_msk);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let grad_scores_raw = mul_scalar_backward(&grad_scores_scl, scale);
    let kt = saved.k.transpose_2d();
    let (grad_q_post, grad_kt) = matmul_backward(&grad_scores_raw, &saved.q, &kt);
    let grad_k_post = grad_kt.transpose_2d();
    let grad_q_pre = rope_backward(&grad_q_post);
    let grad_k_pre = rope_backward(&grad_k_post);
    // v = bit_linear(x, w_v)
    let (grad_x_v, grad_w_v) = bit_linear_backward(&grad_v, x, w_v);
    // q_pre = bit_linear(x, w_q)
    let (grad_x_q, grad_w_q) = bit_linear_backward(&grad_q_pre, x, w_q);
    // k_pre = bit_linear(x, w_k)
    let (grad_x_k, grad_w_k) = bit_linear_backward(&grad_k_pre, x, w_k);
    let grad_x = grad_x_q.add(&grad_x_k).add(&grad_x_v);
    AttentionHeadGrads {
        grad_x,
        grad_w_q,
        grad_w_k,
        grad_w_v,
        grad_w_o,
    }
}

/// Phase 5.a: BitNet SwiGLU FFN forward+save. Same math as
/// `ffn_forward_save` but each matmul against a learnable weight
/// (`w_gate`, `w_up`, `w_down`) becomes a `bit_linear`. Matches the
/// autograd `ffn::ffn` path: x->w_gate / x->w_up share one acts quant
/// internally, and the `gate * up` product is re-quantised before
/// w_down (`bit_linear` does this acts-quant + matmul fusion, so the
/// caller never sees the intermediate).
#[allow(dead_code)]
pub fn ffn_forward_save_bitnet<T>(x: &T, w_gate: &T, w_up: &T, w_down: &T) -> (T, FfnSaved<T>)
where
    T: BitLinear + Silu + Mul,
{
    let gate_pre = x.bit_linear(w_gate);
    let gate = gate_pre.silu();
    let up = x.bit_linear(w_up);
    let h = gate.mul(&up);
    let out = h.bit_linear(w_down);
    (
        out,
        FfnSaved {
            gate_pre,
            gate,
            up,
            h,
        },
    )
}

/// Phase 5.a: BitNet SwiGLU FFN backward. Same chain as
/// `ffn_backward`, with the three matmul_backward calls swapped for
/// bit_linear_backward.
#[allow(dead_code)]
pub fn ffn_backward_bitnet<T>(
    grad_y: &T,
    x: &T,
    saved: &FfnSaved<T>,
    w_gate: &T,
    w_up: &T,
    w_down: &T,
) -> FfnGrads<T>
where
    T: MatMul + Mul + Add + Transpose2D + SiluBackward + QuantiseActsSTE + QuantiseWeightsSTE,
{
    let (grad_h, grad_w_down) = bit_linear_backward(grad_y, &saved.h, w_down);
    let (grad_gate, grad_up) = mul_backward(&grad_h, &saved.gate, &saved.up);
    let grad_gate_pre = silu_backward(&grad_gate, &saved.gate_pre);
    let (grad_x_up, grad_w_up) = bit_linear_backward(&grad_up, x, w_up);
    let (grad_x_gate, grad_w_gate) = bit_linear_backward(&grad_gate_pre, x, w_gate);
    let grad_x = grad_x_gate.add(&grad_x_up);
    FfnGrads {
        grad_x,
        grad_w_gate,
        grad_w_up,
        grad_w_down,
    }
}

/// Phase 5.a: BitNet transformer block forward+save. Composes
/// `attention_head_forward_save_bitnet` per head + `ffn_forward_save_bitnet`
/// with the same residual / rmsnorm structure as `block_forward_save`
/// (chunk 4.5.c).
#[allow(dead_code)]
pub fn block_forward_save_bitnet<T>(
    x: &T,
    heads: &[HeadWeights<T>],
    ffn: &FfnWeights<T>,
    head_dim: usize,
) -> (T, BlockSaved<T>)
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
        + Mul
        + BitLinear,
{
    assert!(
        !heads.is_empty(),
        "block_forward_save_bitnet: at least one head"
    );
    let y1_pre = x.rmsnorm();
    let mut head_saveds: Vec<AttentionHeadSaved<T>> = Vec::with_capacity(heads.len());
    let (head0_out, head0_saved) = attention_head_forward_save_bitnet(
        &y1_pre,
        &heads[0].w_q,
        &heads[0].w_k,
        &heads[0].w_v,
        &heads[0].w_o,
        head_dim,
    );
    head_saveds.push(head0_saved);
    let mut y1 = head0_out;
    for h in &heads[1..] {
        let (out_h, saved_h) =
            attention_head_forward_save_bitnet(&y1_pre, &h.w_q, &h.w_k, &h.w_v, &h.w_o, head_dim);
        y1 = y1.add(&out_h);
        head_saveds.push(saved_h);
    }
    let x1 = x.add(&y1);
    let y2_pre = x1.rmsnorm();
    let (y2, ffn_saved) = ffn_forward_save_bitnet(&y2_pre, &ffn.w_gate, &ffn.w_up, &ffn.w_down);
    let out = x1.add(&y2);
    let saved = BlockSaved {
        y1_pre,
        x1,
        y2_pre,
        head_saveds,
        ffn_saved,
    };
    (out, saved)
}

/// Phase 5.a: BitNet transformer block backward. Same as
/// `block_backward` (chunk 4.5.c) but routes through the bitnet
/// variants of attention and FFN backwards.
#[allow(dead_code)]
pub fn block_backward_bitnet<T>(
    grad_out: &T,
    x: &T,
    saved: &BlockSaved<T>,
    heads: &[HeadWeights<T>],
    ffn: &FfnWeights<T>,
    head_dim: usize,
) -> BlockGrads<T>
where
    T: MatMul
        + Add
        + Mul
        + MulScalar
        + Transpose2D
        + SoftmaxBackward
        + CausalMaskBackward
        + RopeBackward
        + RmsNormBackward
        + SiluBackward
        + QuantiseActsSTE
        + QuantiseWeightsSTE
        + Clone,
{
    assert!(
        !heads.is_empty(),
        "block_backward_bitnet: at least one head"
    );
    let (grad_x1_a, grad_y2) = add_backward(grad_out);
    let ffn_grads = ffn_backward_bitnet(
        &grad_y2,
        &saved.y2_pre,
        &saved.ffn_saved,
        &ffn.w_gate,
        &ffn.w_up,
        &ffn.w_down,
    );
    let grad_x1_b = rmsnorm_backward(&ffn_grads.grad_x, &saved.x1);
    let grad_x1 = grad_x1_a.add(&grad_x1_b);
    let (grad_x_a, grad_y1) = add_backward(&grad_x1);
    let mut head_grads: Vec<HeadWeightGrads<T>> = Vec::with_capacity(heads.len());
    let mut grad_y1_pre: Option<T> = None;
    for (saved_h, h) in saved.head_saveds.iter().zip(heads.iter()) {
        let g = attention_head_backward_bitnet(
            &grad_y1,
            &saved.y1_pre,
            saved_h,
            &h.w_q,
            &h.w_k,
            &h.w_v,
            &h.w_o,
            head_dim,
        );
        grad_y1_pre = Some(match grad_y1_pre {
            None => g.grad_x,
            Some(acc) => acc.add(&g.grad_x),
        });
        head_grads.push(HeadWeightGrads {
            grad_w_q: g.grad_w_q,
            grad_w_k: g.grad_w_k,
            grad_w_v: g.grad_w_v,
            grad_w_o: g.grad_w_o,
        });
    }
    let grad_y1_pre = grad_y1_pre.expect("block_backward_bitnet: at least one head");
    let grad_x_b = rmsnorm_backward(&grad_y1_pre, x);
    let grad_x = grad_x_a.add(&grad_x_b);
    BlockGrads {
        grad_x,
        head_grads,
        grad_w_gate: ffn_grads.grad_w_gate,
        grad_w_up: ffn_grads.grad_w_up,
        grad_w_down: ffn_grads.grad_w_down,
    }
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
pub fn attention_head_inference<T>(x: &T, w_q: &T, w_k: &T, w_v: &T, w_o: &T, head_dim: usize) -> T
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

/// Phase 4 chunk 4.5.a: forward intermediates an attention head needs to
/// retain so its backward closure can reconstruct each op's gradient.
///
/// Why these five and not the others:
/// - `q`, `k` are post-RoPE: matmul backward through `q @ k.T` needs the
///   actual values used in the matmul, and RoPE backward is parameter-
///   free so we never need pre-RoPE values.
/// - `v` is needed for matmul backward through `attn @ v`.
/// - `attn` is needed for both softmax backward (saves output, not
///   input) and matmul backward through `attn @ v`.
/// - `ctx` is needed for matmul backward through `ctx @ w_o`.
///
/// Pre-mask / pre-mul_scalar / pre-softmax tensors are NOT saved -
/// causal_mask backward is shape-only, mul_scalar backward is shape-
/// only, softmax backward saves output.
#[allow(dead_code)]
pub struct AttentionHeadSaved<T> {
    pub q: T,
    pub k: T,
    pub v: T,
    pub attn: T,
    pub ctx: T,
}

/// Phase 4 chunk 4.5.a: gradients an attention head produces. `grad_x`
/// is the gradient flowing back to the head's input (sum of three
/// branches Q, K, V); the four `grad_w_*` tensors are the per-weight
/// gradients in canonical visitor order (Q / K / V / O).
#[allow(dead_code)]
pub struct AttentionHeadGrads<T> {
    pub grad_x: T,
    pub grad_w_q: T,
    pub grad_w_k: T,
    pub grad_w_v: T,
    pub grad_w_o: T,
}

/// Forward pass through one attention head, returning every intermediate
/// the backward needs. Identical math to `attention_head_inference`;
/// the only difference is that this version owns the intermediates in
/// the returned `AttentionHeadSaved<T>` rather than dropping them at
/// the end of the expression. Used as the first half of a hand-traced
/// backward chain - much simpler than building a tape-equivalent for
/// every backend, and acceptable here because the model topology is
/// fixed.
#[allow(dead_code)]
pub fn attention_head_forward_save<T>(
    x: &T,
    w_q: &T,
    w_k: &T,
    w_v: &T,
    w_o: &T,
    head_dim: usize,
) -> (T, AttentionHeadSaved<T>)
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
    let out = ctx.matmul(w_o);
    let saved = AttentionHeadSaved { q, k, v, attn, ctx };
    (out, saved)
}

/// Backward pass through one attention head, hand-traced from the
/// forward in `attention_head_forward_save`. Walks the autograd graph
/// in reverse-creation order and calls each Phase 4 chunk-4.1-4.4
/// helper to chain the gradient through:
///
///     out      = ctx @ w_o
///       -> matmul_backward(grad_out, ctx, w_o)
///     ctx      = attn @ v
///       -> matmul_backward(grad_ctx, attn, v)
///     attn     = softmax(scores_msk)
///       -> softmax_backward(grad_attn, attn)
///     scores_msk = causal_mask(scores_scl)
///       -> causal_mask_backward(grad_scores_msk)
///     scores_scl = scores_raw * scale
///       -> mul_scalar_backward(grad_scores_scl, scale)
///     scores_raw = q @ k.T
///       -> matmul_backward(grad_scores_raw, q, k.T) gives grad_q_post,
///          grad_kt; transpose grad_kt to recover grad_k_post (transpose
///          is its own inverse for rank-2 tensors)
///     q = rope(q_pre); k = rope(k_pre)
///       -> rope_backward(grad_q_post), rope_backward(grad_k_post)
///     q_pre = x @ w_q;  k_pre = x @ w_k;  v = x @ w_v
///       -> three matmul_backwards; each adds its grad_x branch into
///          the running grad_x_total (Q + K + V residual sum at x).
///
/// Only one new trait bound vs `attention_head_forward_save`'s set:
/// `Add` (for the three-branch grad_x sum). Plus the chunk-4.1-4.4
/// backward traits (`SoftmaxBackward`, `CausalMaskBackward`,
/// `RopeBackward`) which are independent of the forward set.
#[allow(dead_code)]
pub fn attention_head_backward<T>(
    grad_out: &T,
    x: &T,
    saved: &AttentionHeadSaved<T>,
    w_q: &T,
    w_k: &T,
    w_v: &T,
    w_o: &T,
    head_dim: usize,
) -> AttentionHeadGrads<T>
where
    T: MatMul + Add + MulScalar + Transpose2D + SoftmaxBackward + CausalMaskBackward + RopeBackward,
{
    // out = ctx @ w_o
    let (grad_ctx, grad_w_o) = matmul_backward(grad_out, &saved.ctx, w_o);
    // ctx = attn @ v
    let (grad_attn, grad_v) = matmul_backward(&grad_ctx, &saved.attn, &saved.v);
    // attn = softmax(scores_msk)
    let grad_scores_msk = softmax_backward(&grad_attn, &saved.attn);
    // scores_msk = causal_mask(scores_scl)
    let grad_scores_scl = causal_mask_backward(&grad_scores_msk);
    // scores_scl = scores_raw * (1 / sqrt(head_dim))
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let grad_scores_raw = mul_scalar_backward(&grad_scores_scl, scale);
    // scores_raw = q @ k.T
    let kt = saved.k.transpose_2d();
    let (grad_q_post, grad_kt) = matmul_backward(&grad_scores_raw, &saved.q, &kt);
    // grad_k_post = (grad_kt).T  - transpose is its own inverse
    let grad_k_post = grad_kt.transpose_2d();
    // q = rope(q_pre); k = rope(k_pre)
    let grad_q_pre = rope_backward(&grad_q_post);
    let grad_k_pre = rope_backward(&grad_k_post);
    // v = x @ w_v
    let (grad_x_v, grad_w_v) = matmul_backward(&grad_v, x, w_v);
    // q_pre = x @ w_q
    let (grad_x_q, grad_w_q) = matmul_backward(&grad_q_pre, x, w_q);
    // k_pre = x @ w_k
    let (grad_x_k, grad_w_k) = matmul_backward(&grad_k_pre, x, w_k);
    // grad_x is the residual sum of the three branches that each
    // consume x: Q matmul, K matmul, V matmul.
    let grad_x = grad_x_q.add(&grad_x_k).add(&grad_x_v);
    AttentionHeadGrads {
        grad_x,
        grad_w_q,
        grad_w_k,
        grad_w_v,
        grad_w_o,
    }
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

/// BitNet absmean-ternary weight quantisation, STE forward (Phase
/// 5.a). Output is `gamma * W_q` where `gamma = mean(|W|)` and
/// `W_q` is in {-1, 0, +1}. Same f32 shape as input. Backward is
/// identity (the STE), so trait does not declare a `_backward`
/// method - callers chain the upstream gradient straight through
/// the pre-quant weight (the gradient flow stays unchanged).
pub trait QuantiseWeightsSTE {
    fn quantise_weights_ste(&self) -> Self;
}

/// BitNet absmax-INT8 per-row activation quantisation, STE forward
/// (Phase 5.a). Output is `(alpha[i] / 127) * x_q[i, j]` where
/// `x_q` lives on the INT8 grid. Backward is identity (STE).
pub trait QuantiseActsSTE {
    fn quantise_acts_ste(&self) -> Self;
}

/// BitNet linear layer forward (Phase 5.a): per-row INT8-STE quantise
/// the activations, ternary-STE quantise the weights, then matmul.
/// This is the math the CPU `Var::matmul` path runs when wrapped in
/// `quantise_acts_ste` + `quantise_weights_ste` (BitLinear in
/// `attention.rs` / `ffn.rs`). The trait surface keeps the same
/// `(&self, &Self) -> Self` shape as `MatMul` so existing helpers
/// can swap one for the other in the bitnet variants.
pub trait BitLinear {
    /// `self` is the activations `[seq, hidden]`; `rhs` is the
    /// learnable weight `[hidden, out]`. Quantises both via STE,
    /// returns the f32 matmul output.
    fn bit_linear(&self, rhs: &Self) -> Self;
}

/// Generic BitLinear backward helper (Phase 5.a). Closed form:
///
///     y = quantise_acts(x) @ quantise_weights(w)
///     STE makes both quants identity in backward, so:
///       grad_x = matmul_backward(grad_y, quantise_acts(x), quantise_weights(w)).0
///       grad_w = matmul_backward(grad_y, quantise_acts(x), quantise_weights(w)).1
///
/// Recomputing the quant during backward is cheaper than carrying
/// extra `[m, n]` saved tensors across the trait surface; quant kernels
/// are small (one or two passes per row / per cell). Trait bounds:
/// `MatMul + Transpose2D + QuantiseActsSTE + QuantiseWeightsSTE`.
#[allow(dead_code)]
pub fn bit_linear_backward<T>(grad_y: &T, x: &T, w: &T) -> (T, T)
where
    T: MatMul + Transpose2D + QuantiseActsSTE + QuantiseWeightsSTE,
{
    let x_eff = x.quantise_acts_ste();
    let w_eff = w.quantise_weights_ste();
    matmul_backward(grad_y, &x_eff, &w_eff)
}

/// Generic BitLinear forward helper that mirrors the chunk-4.x
/// helper-function call style. Equivalent to `x.bit_linear(w)` but
/// reads more naturally inside hand-traced forward code.
#[allow(dead_code)]
pub fn bit_linear<T: BitLinear>(x: &T, w: &T) -> T {
    x.bit_linear(w)
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

    /// Closed-form `matmul_backward` must match the finite-difference
    /// gradient of `loss(A, B) = sum(grad_c * (A @ B))` w.r.t. A and B.
    /// The gradient of that scalar loss is, by construction, exactly the
    /// `(grad_a, grad_b)` the helper produces - so disagreement here would
    /// flag either a math bug in the helper or a bug in `Tensor::matmul`
    /// or `Tensor::transpose_2d`.
    #[test]
    fn matmul_backward_matches_finite_difference_on_cpu() {
        let m = 3usize;
        let k = 4usize;
        let n = 5usize;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.21).cos()).collect();
        let g_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.07).sin() + 0.1).collect();

        let a = Tensor::from_vec(a_data.clone(), vec![m, k]);
        let b = Tensor::from_vec(b_data.clone(), vec![k, n]);
        let grad_c = Tensor::from_vec(g_data.clone(), vec![m, n]);

        let (grad_a, grad_b) = matmul_backward(&grad_c, &a, &b);
        assert_eq!(grad_a.shape, a.shape);
        assert_eq!(grad_b.shape, b.shape);

        // Scalar loss whose gradient is grad_c @ B.T (w.r.t. A) and A.T @ grad_c
        // (w.r.t. B); we check by central finite differences with h=1e-3.
        let loss = |a_v: &[f32], b_v: &[f32]| -> f32 {
            let a_t = Tensor::from_vec(a_v.to_vec(), vec![m, k]);
            let b_t = Tensor::from_vec(b_v.to_vec(), vec![k, n]);
            let c_t = a_t.matmul(&b_t);
            c_t.data.iter().zip(&grad_c.data).map(|(c, g)| c * g).sum()
        };

        let h: f32 = 1e-3;
        // d loss / d A
        for idx in 0..m * k {
            let mut a_plus = a_data.clone();
            a_plus[idx] += h;
            let mut a_minus = a_data.clone();
            a_minus[idx] -= h;
            let fd = (loss(&a_plus, &b_data) - loss(&a_minus, &b_data)) / (2.0 * h);
            let analytic = grad_a.data[idx];
            let abs = (fd - analytic).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * analytic.abs(),
                "grad_a mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
        // d loss / d B
        for idx in 0..k * n {
            let mut b_plus = b_data.clone();
            b_plus[idx] += h;
            let mut b_minus = b_data.clone();
            b_minus[idx] -= h;
            let fd = (loss(&a_data, &b_plus) - loss(&a_data, &b_minus)) / (2.0 * h);
            let analytic = grad_b.data[idx];
            let abs = (fd - analytic).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * analytic.abs(),
                "grad_b mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    /// Headline Phase 4 chunk 4.1 guarantee: the same generic
    /// `matmul_backward` runs on both CPU and CUDA tensors and produces
    /// the same gradients up to f32 round-off. This is what unlocks
    /// future GPU autograd passes - any layer whose backward decomposes
    /// into matmul + transpose calls now works on the GPU with no extra
    /// kernels.
    #[cfg(feature = "cuda")]
    #[test]
    fn matmul_backward_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Prime dimensions to push past tile-aligned shortcuts.
        let m = 17usize;
        let k = 23usize;
        let n = 19usize;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.061).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.097).cos()).collect();
        let g_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.041).sin()).collect();

        let a_cpu = Tensor::from_vec(a_data.clone(), vec![m, k]);
        let b_cpu = Tensor::from_vec(b_data.clone(), vec![k, n]);
        let g_cpu = Tensor::from_vec(g_data.clone(), vec![m, n]);
        let (grad_a_cpu, grad_b_cpu) = matmul_backward(&g_cpu, &a_cpu, &b_cpu);

        let a_gpu = CudaTensor::from_cpu(&a_cpu).expect("H->D failed");
        let b_gpu = CudaTensor::from_cpu(&b_cpu).expect("H->D failed");
        let g_gpu = CudaTensor::from_cpu(&g_cpu).expect("H->D failed");
        let (grad_a_gpu, grad_b_gpu) = matmul_backward(&g_gpu, &a_gpu, &b_gpu);
        let grad_a_back = grad_a_gpu.to_cpu().expect("D->H failed");
        let grad_b_back = grad_b_gpu.to_cpu().expect("D->H failed");

        assert_eq!(grad_a_cpu.shape, grad_a_back.shape);
        assert_eq!(grad_b_cpu.shape, grad_b_back.shape);
        let tol_abs: f32 = 1e-4;
        let tol_rel: f32 = 1e-4;
        for (i, (&c, &g)) in grad_a_cpu.data.iter().zip(&grad_a_back.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= tol_abs + tol_rel * c.abs(),
                "grad_a drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
        for (i, (&c, &g)) in grad_b_cpu.data.iter().zip(&grad_b_back.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= tol_abs + tol_rel * c.abs(),
                "grad_b drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    // ---- Phase 4 chunk 4.2: elementwise backwards ----
    //
    // The CPU finite-difference checks below all share the same recipe:
    // pick small input tensors + a non-trivial upstream gradient
    // `grad_c`, compute analytic gradients via the helper, central-
    // difference the scalar loss `sum(grad_c * forward(...))` cell by
    // cell, assert agreement. Bug in any of the inherent forward ops
    // (`Tensor::silu`, `Tensor::mul`, `Tensor::mul_scalar`) would also
    // show up here, so these double as forward sanity gates.

    #[test]
    fn add_backward_propagates_grad_unchanged_on_cpu() {
        let n = 7usize;
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() + 0.4).collect();
        let grad_c = Tensor::from_vec(g_data.clone(), vec![n]);
        let (grad_a, grad_b) = add_backward(&grad_c);
        assert_eq!(grad_a.shape, grad_c.shape);
        assert_eq!(grad_b.shape, grad_c.shape);
        // Add backward is the identity: each input gets grad_c verbatim.
        for (i, &g) in g_data.iter().enumerate() {
            assert_eq!(grad_a.data[i], g, "grad_a[{i}] not identity");
            assert_eq!(grad_b.data[i], g, "grad_b[{i}] not identity");
        }
    }

    #[test]
    fn mul_backward_matches_finite_difference_on_cpu() {
        let n = 6usize;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.27).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.41).cos()).collect();
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() + 0.2).collect();
        let a = Tensor::from_vec(a_data.clone(), vec![n]);
        let b = Tensor::from_vec(b_data.clone(), vec![n]);
        let grad_c = Tensor::from_vec(g_data.clone(), vec![n]);
        let (grad_a, grad_b) = mul_backward(&grad_c, &a, &b);

        let loss = |a_v: &[f32], b_v: &[f32]| -> f32 {
            a_v.iter()
                .zip(b_v)
                .zip(&g_data)
                .map(|((x, y), g)| g * (x * y))
                .sum()
        };
        let h: f32 = 1e-3;
        for idx in 0..n {
            let (mut ap, mut am) = (a_data.clone(), a_data.clone());
            ap[idx] += h;
            am[idx] -= h;
            let fd = (loss(&ap, &b_data) - loss(&am, &b_data)) / (2.0 * h);
            let analytic = grad_a.data[idx];
            assert!(
                (fd - analytic).abs() <= 1e-3 + 1e-3 * analytic.abs(),
                "grad_a mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
            let (mut bp, mut bm) = (b_data.clone(), b_data.clone());
            bp[idx] += h;
            bm[idx] -= h;
            let fd = (loss(&a_data, &bp) - loss(&a_data, &bm)) / (2.0 * h);
            let analytic = grad_b.data[idx];
            assert!(
                (fd - analytic).abs() <= 1e-3 + 1e-3 * analytic.abs(),
                "grad_b mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    #[test]
    fn mul_scalar_backward_matches_finite_difference_on_cpu() {
        let n = 5usize;
        let s_scalar: f32 = 2.5;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.19).sin()).collect();
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos() + 0.3).collect();
        let grad_c = Tensor::from_vec(g_data.clone(), vec![n]);
        let grad_a = mul_scalar_backward(&grad_c, s_scalar);

        let loss = |a_v: &[f32]| -> f32 {
            a_v.iter()
                .zip(&g_data)
                .map(|(x, g)| g * (x * s_scalar))
                .sum()
        };
        let h: f32 = 1e-3;
        for idx in 0..n {
            let (mut ap, mut am) = (a_data.clone(), a_data.clone());
            ap[idx] += h;
            am[idx] -= h;
            let fd = (loss(&ap) - loss(&am)) / (2.0 * h);
            let analytic = grad_a.data[idx];
            assert!(
                (fd - analytic).abs() <= 1e-3 + 1e-3 * analytic.abs(),
                "grad_a mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    #[test]
    fn silu_backward_matches_finite_difference_on_cpu() {
        let n = 8usize;
        // Cover positive, near-zero, and negative regions; silu's curve
        // is most non-linear around zero so a finite-diff bug is most
        // likely to show up there.
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 3.5) * 0.6).collect();
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).sin() + 0.5).collect();
        let x = Tensor::from_vec(x_data.clone(), vec![n]);
        let grad_y = Tensor::from_vec(g_data.clone(), vec![n]);
        let grad_x = silu_backward(&grad_y, &x);

        let loss = |x_v: &[f32]| -> f32 {
            let xt = Tensor::from_vec(x_v.to_vec(), vec![n]);
            let yt = xt.silu();
            yt.data.iter().zip(&g_data).map(|(y, g)| g * y).sum()
        };
        let h: f32 = 1e-3;
        for idx in 0..n {
            let (mut xp, mut xm) = (x_data.clone(), x_data.clone());
            xp[idx] += h;
            xm[idx] -= h;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * h);
            let analytic = grad_x.data[idx];
            assert!(
                (fd - analytic).abs() <= 1e-3 + 1e-3 * analytic.abs(),
                "silu_backward mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    /// Cross-backend agreement: the same generic chunk-4.2 helpers
    /// produce matching gradients on `Tensor` and `CudaTensor`. The
    /// `add_backward` arm of this test is also the only place we
    /// exercise `CudaTensor::clone()` (the device-to-device memcpy
    /// path), so a regression in `Clone` would surface here.
    #[cfg(feature = "cuda")]
    #[test]
    fn elementwise_backwards_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let n = 23usize;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.061).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.097).cos()).collect();
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.041).sin() + 0.2).collect();
        let s_scalar: f32 = -1.75;

        let a_cpu = Tensor::from_vec(a_data.clone(), vec![n]);
        let b_cpu = Tensor::from_vec(b_data.clone(), vec![n]);
        let g_cpu = Tensor::from_vec(g_data.clone(), vec![n]);
        let a_gpu = CudaTensor::from_cpu(&a_cpu).expect("H->D");
        let b_gpu = CudaTensor::from_cpu(&b_cpu).expect("H->D");
        let g_gpu = CudaTensor::from_cpu(&g_cpu).expect("H->D");

        let assert_close = |label: &str, cpu_t: &Tensor, gpu_t: &Tensor| {
            assert_eq!(cpu_t.shape, gpu_t.shape, "{label}: shape diverged");
            for (i, (&c, &g)) in cpu_t.data.iter().zip(&gpu_t.data).enumerate() {
                let abs = (c - g).abs();
                assert!(
                    abs <= 1e-4 + 1e-4 * c.abs(),
                    "{label} drift at idx {i}: cpu = {c}, cuda = {g}"
                );
            }
        };

        // add_backward: hits CudaTensor::clone path twice.
        let (cpu_a1, cpu_a2) = add_backward(&g_cpu);
        let (gpu_a1, gpu_a2) = add_backward(&g_gpu);
        assert_close("add_backward.0", &cpu_a1, &gpu_a1.to_cpu().expect("D->H"));
        assert_close("add_backward.1", &cpu_a2, &gpu_a2.to_cpu().expect("D->H"));

        // mul_backward: tests Mul trait impl on both backends.
        let (cpu_ga, cpu_gb) = mul_backward(&g_cpu, &a_cpu, &b_cpu);
        let (gpu_ga, gpu_gb) = mul_backward(&g_gpu, &a_gpu, &b_gpu);
        assert_close("mul_backward.0", &cpu_ga, &gpu_ga.to_cpu().expect("D->H"));
        assert_close("mul_backward.1", &cpu_gb, &gpu_gb.to_cpu().expect("D->H"));

        // mul_scalar_backward: tests MulScalar trait impl on both backends.
        let cpu_ms = mul_scalar_backward(&g_cpu, s_scalar);
        let gpu_ms = mul_scalar_backward(&g_gpu, s_scalar)
            .to_cpu()
            .expect("D->H");
        assert_close("mul_scalar_backward", &cpu_ms, &gpu_ms);

        // silu_backward: the headline cross-backend test for the new
        // fused `silu_backward_f32` kernel. Saved input `x` covers the
        // sigmoid's non-linear region around zero on purpose.
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 11.0) * 0.4).collect();
        let x_cpu = Tensor::from_vec(x_data.clone(), vec![n]);
        let x_gpu = CudaTensor::from_cpu(&x_cpu).expect("H->D");
        let cpu_sb = silu_backward(&g_cpu, &x_cpu);
        let gpu_sb = silu_backward(&g_gpu, &x_gpu).to_cpu().expect("D->H");
        assert_close("silu_backward", &cpu_sb, &gpu_sb);
    }

    // ---- Phase 4 chunk 4.3: softmax + causal-mask backwards ----

    /// CPU finite-diff for softmax backward. The Jacobian formula is
    /// `J = diag(s) - s s^T`; central differences of
    /// `loss(x) = sum(grad_y * softmax(x))` cell by cell catch any
    /// sign / index error in either the helper or `Tensor::softmax`.
    /// Rows are independent so a 3x4 case is sufficient coverage.
    #[test]
    fn softmax_backward_matches_finite_difference_on_cpu() {
        let m = 3usize;
        let n = 4usize;
        // x-values that span a full sigmoid range so non-trivial
        // softmax outputs (no degenerate one-hot) feed the Jacobian.
        let x_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.37).sin()).collect();
        let g_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.21).cos() + 0.3).collect();
        let x = Tensor::from_vec(x_data.clone(), vec![m, n]);
        let s_out = x.softmax();
        let grad_y = Tensor::from_vec(g_data.clone(), vec![m, n]);
        let grad_x = softmax_backward(&grad_y, &s_out);
        assert_eq!(grad_x.shape, x.shape);

        let loss = |x_v: &[f32]| -> f32 {
            let xt = Tensor::from_vec(x_v.to_vec(), vec![m, n]);
            let st = xt.softmax();
            st.data.iter().zip(&g_data).map(|(s, g)| g * s).sum()
        };
        let h: f32 = 1e-3;
        for idx in 0..m * n {
            let (mut xp, mut xm) = (x_data.clone(), x_data.clone());
            xp[idx] += h;
            xm[idx] -= h;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * h);
            let analytic = grad_x.data[idx];
            assert!(
                (fd - analytic).abs() <= 1e-3 + 1e-3 * analytic.abs(),
                "softmax_backward mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    /// Causal mask backward is structurally trivial (identity on the
    /// lower triangle, zero above) but finite-diff is unusable because
    /// the forward writes `-inf` into the upper triangle, which would
    /// make `sum(grad_y * forward(x))` itself non-finite. Direct
    /// structural check: every cell either passes through unchanged or
    /// is zeroed.
    #[test]
    fn causal_mask_backward_zeros_upper_triangle_on_cpu() {
        let n = 5usize;
        let g_data: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.17).sin() + 0.2).collect();
        let grad_y = Tensor::from_vec(g_data.clone(), vec![n, n]);
        let grad_x = causal_mask_backward(&grad_y);
        assert_eq!(grad_x.shape, vec![n, n]);
        for i in 0..n {
            for j in 0..n {
                let want = if j > i { 0.0 } else { g_data[i * n + j] };
                assert_eq!(
                    grad_x.data[i * n + j],
                    want,
                    "causal_mask_backward at ({i},{j}): want {want}, got {}",
                    grad_x.data[i * n + j]
                );
            }
        }
    }

    /// Cross-backend agreement for both chunk-4.3 backwards. Softmax
    /// backward is the headline arm: the new `softmax_backward_row_f32`
    /// kernel is exercised here for the first time. `s_out` for the
    /// softmax test is computed via `Tensor::softmax` on CPU and then
    /// copied to device, mirroring how a real autograd path would
    /// already have the saved forward output on whichever backend
    /// produced it.
    #[cfg(feature = "cuda")]
    #[test]
    fn softmax_and_causal_mask_backwards_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Prime dimensions catch off-by-one errors in row-stride math
        // that square shapes hide. m=11 rows, n=13 cols for softmax.
        let m = 11usize;
        let n = 13usize;
        let x_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.041).sin()).collect();
        let g_data_sm: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.097).cos() + 0.1).collect();
        let x_cpu = Tensor::from_vec(x_data.clone(), vec![m, n]);
        let s_out_cpu = x_cpu.softmax();
        let s_out_gpu = CudaTensor::from_cpu(&s_out_cpu).expect("H->D s_out");
        let g_cpu_sm = Tensor::from_vec(g_data_sm.clone(), vec![m, n]);
        let g_gpu_sm = CudaTensor::from_cpu(&g_cpu_sm).expect("H->D g_sm");

        let cpu_sm = softmax_backward(&g_cpu_sm, &s_out_cpu);
        let gpu_sm = softmax_backward(&g_gpu_sm, &s_out_gpu)
            .to_cpu()
            .expect("D->H softmax_backward");

        assert_eq!(cpu_sm.shape, gpu_sm.shape);
        for (i, (&c, &g)) in cpu_sm.data.iter().zip(&gpu_sm.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-4 + 1e-4 * c.abs(),
                "softmax_backward drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }

        // Causal mask: square seq=17 (prime; covers non-multiple-of-16
        // tile fall-off in the 2-D launch shape).
        let s = 17usize;
        let g_data_cm: Vec<f32> = (0..s * s)
            .map(|i| (i as f32 * 0.061).sin() + 0.05)
            .collect();
        let g_cpu_cm = Tensor::from_vec(g_data_cm.clone(), vec![s, s]);
        let g_gpu_cm = CudaTensor::from_cpu(&g_cpu_cm).expect("H->D g_cm");
        let cpu_cm = causal_mask_backward(&g_cpu_cm);
        let gpu_cm = causal_mask_backward(&g_gpu_cm)
            .to_cpu()
            .expect("D->H causal_mask_backward");
        assert_eq!(cpu_cm.shape, gpu_cm.shape);
        for (i, (&c, &g)) in cpu_cm.data.iter().zip(&gpu_cm.data).enumerate() {
            // Zeros and identity-flow values both: identical bit
            // pattern expected (no f32 math involved on either side).
            assert_eq!(
                c, g,
                "causal_mask_backward bit-mismatch at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    // ---- Phase 4 chunk 4.4: rmsnorm + rope backwards ----

    /// CPU finite-diff for RMSNorm backward. The per-row JVP couples
    /// every cell of the row through the shared `rms_i` scalar, so a
    /// 3x4 case exercises the full coupling pattern and any bug
    /// in the `inv_rms / dot / factor` arithmetic surfaces here.
    #[test]
    fn rmsnorm_backward_matches_finite_difference_on_cpu() {
        let m = 3usize;
        let n = 4usize;
        // Avoid all-zero rows (RMSNorm is well-defined everywhere
        // because of EPS, but finite-diff stability prefers
        // away-from-zero magnitudes).
        let x_data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.43 + 0.5).sin() * 1.5)
            .collect();
        let g_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.31).cos() + 0.4).collect();
        let x = Tensor::from_vec(x_data.clone(), vec![m, n]);
        let grad_y = Tensor::from_vec(g_data.clone(), vec![m, n]);
        let grad_x = rmsnorm_backward(&grad_y, &x);
        assert_eq!(grad_x.shape, x.shape);

        let loss = |x_v: &[f32]| -> f32 {
            let xt = Tensor::from_vec(x_v.to_vec(), vec![m, n]);
            let yt = xt.rmsnorm();
            yt.data.iter().zip(&g_data).map(|(y, g)| g * y).sum()
        };
        let h: f32 = 1e-3;
        for idx in 0..m * n {
            let (mut xp, mut xm) = (x_data.clone(), x_data.clone());
            xp[idx] += h;
            xm[idx] -= h;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * h);
            let analytic = grad_x.data[idx];
            assert!(
                (fd - analytic).abs() <= 2e-3 + 2e-3 * analytic.abs(),
                "rmsnorm_backward mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    /// CPU finite-diff for RoPE backward. RoPE is parameter-free and
    /// per-pair orthogonal, so a small `[seq=4, head_dim=4]` case
    /// exercises every pair index (i=0, i=1) and several positions
    /// (the high-frequency pair at i=0 dominates the rotation).
    #[test]
    fn rope_backward_matches_finite_difference_on_cpu() {
        let seq = 4usize;
        let head_dim = 4usize;
        let n = seq * head_dim;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.29).sin()).collect();
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).cos() + 0.3).collect();
        let grad_y = Tensor::from_vec(g_data.clone(), vec![seq, head_dim]);
        let grad_x = rope_backward(&grad_y);
        assert_eq!(grad_x.shape, vec![seq, head_dim]);

        let loss = |x_v: &[f32]| -> f32 {
            let xt = Tensor::from_vec(x_v.to_vec(), vec![seq, head_dim]);
            let yt = xt.rope();
            yt.data.iter().zip(&g_data).map(|(y, g)| g * y).sum()
        };
        let h: f32 = 1e-3;
        for idx in 0..n {
            let (mut xp, mut xm) = (x_data.clone(), x_data.clone());
            xp[idx] += h;
            xm[idx] -= h;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * h);
            let analytic = grad_x.data[idx];
            assert!(
                (fd - analytic).abs() <= 1e-3 + 1e-3 * analytic.abs(),
                "rope_backward mismatch at idx {idx}: fd = {fd}, analytic = {analytic}"
            );
        }
    }

    /// Cross-backend agreement for both chunk-4.4 backwards. RMSNorm
    /// backward has the trickier arithmetic (three scalar terms per
    /// row coupled across cells); RoPE backward has the trickier
    /// indexing (interleaved `2i / 2i+1` pairs). Both check at prime
    /// shapes that catch off-by-one bugs in the GPU launch math.
    #[cfg(feature = "cuda")]
    #[test]
    fn rmsnorm_and_rope_backwards_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }

        // RMSNorm: prime m=11 rows, n=13 cols (matches chunk-4.3
        // softmax test for symmetry).
        let m = 11usize;
        let n = 13usize;
        let x_data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.041 + 0.3).sin() * 1.2)
            .collect();
        let g_data_rn: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.097).cos() + 0.2).collect();
        let x_cpu = Tensor::from_vec(x_data.clone(), vec![m, n]);
        let x_gpu = CudaTensor::from_cpu(&x_cpu).expect("H->D x");
        let g_cpu_rn = Tensor::from_vec(g_data_rn.clone(), vec![m, n]);
        let g_gpu_rn = CudaTensor::from_cpu(&g_cpu_rn).expect("H->D g_rn");

        let cpu_rn = rmsnorm_backward(&g_cpu_rn, &x_cpu);
        let gpu_rn = rmsnorm_backward(&g_gpu_rn, &x_gpu)
            .to_cpu()
            .expect("D->H rmsnorm_backward");
        assert_eq!(cpu_rn.shape, gpu_rn.shape);
        for (i, (&c, &g)) in cpu_rn.data.iter().zip(&gpu_rn.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-4 + 1e-4 * c.abs(),
                "rmsnorm_backward drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }

        // RoPE: prime seq=13, even head_dim=18 (head_dim must be
        // even; 18 = 2*9 forces 9 distinct pairs which spans the
        // full base-10000 frequency range cleanly).
        let seq = 13usize;
        let head_dim = 18usize;
        let n_rope = seq * head_dim;
        let g_data_rp: Vec<f32> = (0..n_rope)
            .map(|i| (i as f32 * 0.061).sin() + 0.1)
            .collect();
        let g_cpu_rp = Tensor::from_vec(g_data_rp.clone(), vec![seq, head_dim]);
        let g_gpu_rp = CudaTensor::from_cpu(&g_cpu_rp).expect("H->D g_rp");
        let cpu_rp = rope_backward(&g_cpu_rp);
        let gpu_rp = rope_backward(&g_gpu_rp)
            .to_cpu()
            .expect("D->H rope_backward");
        assert_eq!(cpu_rp.shape, gpu_rp.shape);
        // Looser tolerance than the rmsnorm arm: cos/sin/powf differ
        // in the last few mantissa bits between CPU libm and CUDA's
        // intrinsics, same trade-off the forward-RoPE test absorbs.
        for (i, (&c, &g)) in cpu_rp.data.iter().zip(&gpu_rp.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "rope_backward drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    // ---- Phase 4 chunk 4.5.a: full attention-head forward+backward ----

    /// Phase 5.a headline correctness check: the bitnet attention head
    /// (chunk-5.a `attention_head_*_bitnet` helpers) produces the same
    /// gradients as the autograd Var path with explicit
    /// `quantise_acts_ste` + `quantise_weights_ste` wrappers. STE makes
    /// both quants identity in backward, so swapping `matmul` for
    /// `bit_linear` should leave the math invariant; this test
    /// confirms the wiring matches the autograd `attention::head_output`
    /// recipe in `attention.rs`.
    #[test]
    fn attention_head_backward_bitnet_matches_autograd_ground_truth() {
        use crate::autograd::{Tape, Var};
        let head_dim = 4usize;
        let seq = 6usize;
        let hidden = 12usize;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.071).sin() * 0.7)
            .collect();
        let w_q_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.103).cos() * 0.4)
            .collect();
        let w_k_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.137).sin() * 0.4)
            .collect();
        let w_v_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.181 + 0.2).cos() * 0.4)
            .collect();
        let w_o_data: Vec<f32> = (0..head_dim * hidden)
            .map(|i| (i as f32 * 0.211).sin() * 0.4)
            .collect();

        // Path A: Var-based autograd with explicit quant wrappers,
        // matching the head_output recipe in attention.rs:80-110.
        let tape = Tape::new();
        let x_var = Var::leaf(&tape, Tensor::from_vec(x_data.clone(), vec![seq, hidden]));
        let w_q_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_q_data.clone(), vec![hidden, head_dim]),
        );
        let w_k_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_k_data.clone(), vec![hidden, head_dim]),
        );
        let w_v_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_v_data.clone(), vec![hidden, head_dim]),
        );
        let w_o_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_o_data.clone(), vec![head_dim, hidden]),
        );
        let x_eff = x_var.quantise_acts_ste();
        let q_v = x_eff.matmul(w_q_var.quantise_weights_ste()).rope();
        let k_v = x_eff.matmul(w_k_var.quantise_weights_ste()).rope();
        let v_v = x_eff.matmul(w_v_var.quantise_weights_ste());
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let scores_v = q_v
            .matmul(k_v.transpose_2d())
            .mul_scalar(scale)
            .causal_mask();
        let attn_v = scores_v.softmax();
        let ctx_v = attn_v.matmul(v_v);
        let out_v = ctx_v
            .quantise_acts_ste()
            .matmul(w_o_var.quantise_weights_ste());
        tape.backward(out_v.id);
        let grad_x_truth = x_var.grad();
        let grad_w_q_truth = w_q_var.grad();
        let grad_w_k_truth = w_k_var.grad();
        let grad_w_v_truth = w_v_var.grad();
        let grad_w_o_truth = w_o_var.grad();

        // Path B: chunk-5.a bitnet helpers.
        let x = Tensor::from_vec(x_data, vec![seq, hidden]);
        let w_q = Tensor::from_vec(w_q_data, vec![hidden, head_dim]);
        let w_k = Tensor::from_vec(w_k_data, vec![hidden, head_dim]);
        let w_v = Tensor::from_vec(w_v_data, vec![hidden, head_dim]);
        let w_o = Tensor::from_vec(w_o_data, vec![head_dim, hidden]);
        let (out, saved) = attention_head_forward_save_bitnet(&x, &w_q, &w_k, &w_v, &w_o, head_dim);
        let grad_out = Tensor::ones(out.shape.clone());
        let grads =
            attention_head_backward_bitnet(&grad_out, &x, &saved, &w_q, &w_k, &w_v, &w_o, head_dim);

        let cmp = |label: &str, truth: &Tensor, helper: &Tensor| {
            assert_eq!(truth.shape, helper.shape, "{label}: shape diverged");
            for (i, (&t, &h)) in truth.data.iter().zip(&helper.data).enumerate() {
                assert!(
                    (t - h).abs() <= 1e-4 + 1e-4 * t.abs(),
                    "{label} mismatch at idx {i}: autograd = {t}, bitnet = {h}"
                );
            }
        };
        cmp("grad_x", &grad_x_truth, &grads.grad_x);
        cmp("grad_w_q", &grad_w_q_truth, &grads.grad_w_q);
        cmp("grad_w_k", &grad_w_k_truth, &grads.grad_w_k);
        cmp("grad_w_v", &grad_w_v_truth, &grads.grad_w_v);
        cmp("grad_w_o", &grad_w_o_truth, &grads.grad_w_o);
    }

    /// Headline correctness check for `attention_head_forward_save` +
    /// `attention_head_backward`. Builds the same head two different
    /// ways:
    ///
    ///   path A: existing tape-based `Var` autograd (the project's
    ///           pre-Phase-4 source of truth for gradients);
    ///   path B: hand-traced backward chain using the chunk-4.1-4.4
    ///           generic helpers, run on `Tensor`.
    ///
    /// Asserts every gradient cell agrees within a tight tolerance.
    /// Path A's `Tape::backward` seeds the output gradient with ones,
    /// so path B passes `Tensor::ones(out.shape)` as `grad_out` for a
    /// like-for-like compare. Any disagreement here would imply a math
    /// error in the new helpers, in one of the chunk-4.1-4.4 backward
    /// kernels, or in the saved-intermediate selection in
    /// `AttentionHeadSaved`.
    #[test]
    fn attention_head_backward_matches_autograd_ground_truth() {
        use crate::autograd::{Tape, Var};
        let head_dim = 4usize;
        let seq = 6usize;
        let hidden = 12usize;

        // Deterministic non-trivial inputs. Mixing sin/cos and offsets
        // avoids degenerate (all-zero, all-one) shapes that hide bugs.
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.071).sin() * 0.7)
            .collect();
        let w_q_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.103).cos() * 0.4)
            .collect();
        let w_k_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.137).sin() * 0.4)
            .collect();
        let w_v_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.181 + 0.2).cos() * 0.4)
            .collect();
        let w_o_data: Vec<f32> = (0..head_dim * hidden)
            .map(|i| (i as f32 * 0.211).sin() * 0.4)
            .collect();

        // ---- Path A: Var-based autograd (ground truth). ----
        let tape = Tape::new();
        let x_var = Var::leaf(&tape, Tensor::from_vec(x_data.clone(), vec![seq, hidden]));
        let w_q_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_q_data.clone(), vec![hidden, head_dim]),
        );
        let w_k_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_k_data.clone(), vec![hidden, head_dim]),
        );
        let w_v_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_v_data.clone(), vec![hidden, head_dim]),
        );
        let w_o_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_o_data.clone(), vec![head_dim, hidden]),
        );

        let q_v = x_var.matmul(w_q_var).rope();
        let k_v = x_var.matmul(w_k_var).rope();
        let v_v = x_var.matmul(w_v_var);
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let scores_v = q_v
            .matmul(k_v.transpose_2d())
            .mul_scalar(scale)
            .causal_mask();
        let attn_v = scores_v.softmax();
        let ctx_v = attn_v.matmul(v_v);
        let out_v = ctx_v.matmul(w_o_var);
        tape.backward(out_v.id);

        let grad_x_truth = x_var.grad();
        let grad_w_q_truth = w_q_var.grad();
        let grad_w_k_truth = w_k_var.grad();
        let grad_w_v_truth = w_v_var.grad();
        let grad_w_o_truth = w_o_var.grad();

        // ---- Path B: hand-traced chunk-4.5.a backward. ----
        let x = Tensor::from_vec(x_data, vec![seq, hidden]);
        let w_q = Tensor::from_vec(w_q_data, vec![hidden, head_dim]);
        let w_k = Tensor::from_vec(w_k_data, vec![hidden, head_dim]);
        let w_v = Tensor::from_vec(w_v_data, vec![hidden, head_dim]);
        let w_o = Tensor::from_vec(w_o_data, vec![head_dim, hidden]);
        let (out, saved) = attention_head_forward_save(&x, &w_q, &w_k, &w_v, &w_o, head_dim);
        // Forward outputs must match within tight FP tolerance (the
        // forward path is identical to `attention_head_inference`,
        // which is already cross-checked, so this is a sanity belt).
        let out_truth = out_v.value();
        assert_eq!(out.shape, out_truth.shape);
        for (i, (&a, &b)) in out.data.iter().zip(&out_truth.data).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 + 1e-5 * a.abs(),
                "forward mismatch at idx {i}: helper = {a}, autograd = {b}"
            );
        }
        let grad_out = Tensor::ones(out.shape.clone());
        let grads =
            attention_head_backward(&grad_out, &x, &saved, &w_q, &w_k, &w_v, &w_o, head_dim);

        // ---- Compare. ----
        let cmp = |label: &str, truth: &Tensor, helper: &Tensor| {
            assert_eq!(truth.shape, helper.shape, "{label}: shape diverged");
            for (i, (&t, &h)) in truth.data.iter().zip(&helper.data).enumerate() {
                let abs = (t - h).abs();
                assert!(
                    abs <= 1e-4 + 1e-4 * t.abs(),
                    "{label} mismatch at idx {i}: autograd = {t}, helper = {h}"
                );
            }
        };
        cmp("grad_x", &grad_x_truth, &grads.grad_x);
        cmp("grad_w_q", &grad_w_q_truth, &grads.grad_w_q);
        cmp("grad_w_k", &grad_w_k_truth, &grads.grad_w_k);
        cmp("grad_w_v", &grad_w_v_truth, &grads.grad_w_v);
        cmp("grad_w_o", &grad_w_o_truth, &grads.grad_w_o);
    }

    /// The headline cross-backend test for chunk 4.5.a: the same
    /// hand-traced attention-head backward runs end-to-end on `Tensor`
    /// and `CudaTensor` over identical inputs, and every gradient cell
    /// agrees within FP tolerance. This is the first test that
    /// exercises **all** chunk 4.1-4.4 backward kernels in one chain
    /// on the GPU - per-op tests pass each kernel through a single
    /// invocation; this one composes them.
    #[cfg(feature = "cuda")]
    #[test]
    fn attention_head_backward_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Realistic-ish v0.13 attention shapes (just smaller seq for
        // test speed): seq=8, hidden=24, head_dim=8.
        let head_dim = 8usize;
        let seq = 8usize;
        let hidden = 24usize;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.041).sin() * 0.6)
            .collect();
        let w_q_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.097).cos() * 0.3)
            .collect();
        let w_k_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.131).sin() * 0.3)
            .collect();
        let w_v_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.169).cos() * 0.3)
            .collect();
        let w_o_data: Vec<f32> = (0..head_dim * hidden)
            .map(|i| (i as f32 * 0.193).sin() * 0.3)
            .collect();
        let g_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.061).cos() + 0.1)
            .collect();

        let x_cpu = Tensor::from_vec(x_data.clone(), vec![seq, hidden]);
        let w_q_cpu = Tensor::from_vec(w_q_data.clone(), vec![hidden, head_dim]);
        let w_k_cpu = Tensor::from_vec(w_k_data.clone(), vec![hidden, head_dim]);
        let w_v_cpu = Tensor::from_vec(w_v_data.clone(), vec![hidden, head_dim]);
        let w_o_cpu = Tensor::from_vec(w_o_data.clone(), vec![head_dim, hidden]);
        let g_cpu = Tensor::from_vec(g_data.clone(), vec![seq, hidden]);

        let (out_cpu, saved_cpu) =
            attention_head_forward_save(&x_cpu, &w_q_cpu, &w_k_cpu, &w_v_cpu, &w_o_cpu, head_dim);
        let grads_cpu = attention_head_backward(
            &g_cpu, &x_cpu, &saved_cpu, &w_q_cpu, &w_k_cpu, &w_v_cpu, &w_o_cpu, head_dim,
        );

        let x_gpu = CudaTensor::from_cpu(&x_cpu).expect("H->D x");
        let w_q_gpu = CudaTensor::from_cpu(&w_q_cpu).expect("H->D w_q");
        let w_k_gpu = CudaTensor::from_cpu(&w_k_cpu).expect("H->D w_k");
        let w_v_gpu = CudaTensor::from_cpu(&w_v_cpu).expect("H->D w_v");
        let w_o_gpu = CudaTensor::from_cpu(&w_o_cpu).expect("H->D w_o");
        let g_gpu = CudaTensor::from_cpu(&g_cpu).expect("H->D g");

        let (out_gpu, saved_gpu) =
            attention_head_forward_save(&x_gpu, &w_q_gpu, &w_k_gpu, &w_v_gpu, &w_o_gpu, head_dim);
        let grads_gpu = attention_head_backward(
            &g_gpu, &x_gpu, &saved_gpu, &w_q_gpu, &w_k_gpu, &w_v_gpu, &w_o_gpu, head_dim,
        );

        let assert_close = |label: &str, cpu_t: &Tensor, gpu_t_dev: &CudaTensor, tol: f32| {
            let gpu_t = gpu_t_dev.to_cpu().expect("D->H");
            assert_eq!(cpu_t.shape, gpu_t.shape, "{label}: shape diverged");
            for (i, (&c, &g)) in cpu_t.data.iter().zip(&gpu_t.data).enumerate() {
                let abs = (c - g).abs();
                assert!(
                    abs <= tol + tol * c.abs(),
                    "{label} drift at idx {i}: cpu = {c}, cuda = {g}"
                );
            }
        };
        // Forward sanity: identical inputs through identical math
        // should match within the per-block tolerance from chunk 2.4.
        assert_close("forward output", &out_cpu, &out_gpu, 5e-3);
        // Backward: looser still, because we chain 5 matmuls + softmax
        // + 2 ropes + transpose + sum-of-three-branches, with each step
        // accumulating drift from the parallel-reduction f32 sums in
        // cuBLAS / NVRTC.
        assert_close("grad_x", &grads_cpu.grad_x, &grads_gpu.grad_x, 1e-2);
        assert_close("grad_w_q", &grads_cpu.grad_w_q, &grads_gpu.grad_w_q, 1e-2);
        assert_close("grad_w_k", &grads_cpu.grad_w_k, &grads_gpu.grad_w_k, 1e-2);
        assert_close("grad_w_v", &grads_cpu.grad_w_v, &grads_gpu.grad_w_v, 1e-2);
        assert_close("grad_w_o", &grads_cpu.grad_w_o, &grads_gpu.grad_w_o, 1e-2);
    }

    // ---- Phase 4 chunk 4.5.b: full SwiGLU FFN forward+backward ----

    /// Mirror of `attention_head_backward_matches_autograd_ground_truth`
    /// for the FFN. Path A uses the project's `Var`-based autograd
    /// (matmul + silu + mul through the same algebraic chain); path B
    /// uses the new chunk-4.5.b helpers on plain `Tensor`. Both seed
    /// the output gradient with ones (matching `Tape::backward`'s
    /// default seed). Asserts every gradient cell agrees within
    /// `1e-4 + 1e-4 * |val|` - any disagreement implies a math error
    /// in `mul_backward`, `silu_backward`, the chunk-4.1 matmul
    /// backward, or the wiring in `ffn_backward`.
    #[test]
    fn ffn_backward_matches_autograd_ground_truth() {
        use crate::autograd::{Tape, Var};
        let seq = 5usize;
        let hidden = 8usize;
        let ffn = 16usize;

        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.073).sin() * 0.6)
            .collect();
        let w_gate_data: Vec<f32> = (0..hidden * ffn)
            .map(|i| (i as f32 * 0.107).cos() * 0.4)
            .collect();
        let w_up_data: Vec<f32> = (0..hidden * ffn)
            .map(|i| (i as f32 * 0.139).sin() * 0.4)
            .collect();
        let w_down_data: Vec<f32> = (0..ffn * hidden)
            .map(|i| (i as f32 * 0.173).cos() * 0.4)
            .collect();

        // ---- Path A: Var-based autograd. ----
        let tape = Tape::new();
        let x_var = Var::leaf(&tape, Tensor::from_vec(x_data.clone(), vec![seq, hidden]));
        let w_gate_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_gate_data.clone(), vec![hidden, ffn]),
        );
        let w_up_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_up_data.clone(), vec![hidden, ffn]),
        );
        let w_down_var = Var::leaf(
            &tape,
            Tensor::from_vec(w_down_data.clone(), vec![ffn, hidden]),
        );
        let gate_v = x_var.matmul(w_gate_var).silu();
        let up_v = x_var.matmul(w_up_var);
        let h_v = gate_v.mul(up_v);
        let out_v = h_v.matmul(w_down_var);
        tape.backward(out_v.id);
        let grad_x_truth = x_var.grad();
        let grad_w_gate_truth = w_gate_var.grad();
        let grad_w_up_truth = w_up_var.grad();
        let grad_w_down_truth = w_down_var.grad();

        // ---- Path B: chunk-4.5.b helpers. ----
        let x = Tensor::from_vec(x_data, vec![seq, hidden]);
        let w_gate = Tensor::from_vec(w_gate_data, vec![hidden, ffn]);
        let w_up = Tensor::from_vec(w_up_data, vec![hidden, ffn]);
        let w_down = Tensor::from_vec(w_down_data, vec![ffn, hidden]);
        let (out, saved) = ffn_forward_save(&x, &w_gate, &w_up, &w_down);
        let grad_out = Tensor::ones(out.shape.clone());
        let grads = ffn_backward(&grad_out, &x, &saved, &w_gate, &w_up, &w_down);

        // Forward sanity (the helper's forward path is identical to
        // `ffn_inference`, which is already cross-checked, but the
        // saved-intermediate refactor is new this chunk so the belt
        // matters).
        let out_truth = out_v.value();
        for (i, (&a, &b)) in out.data.iter().zip(&out_truth.data).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 + 1e-5 * a.abs(),
                "forward mismatch at idx {i}: helper = {a}, autograd = {b}"
            );
        }
        let cmp = |label: &str, truth: &Tensor, helper: &Tensor| {
            assert_eq!(truth.shape, helper.shape, "{label}: shape diverged");
            for (i, (&t, &h)) in truth.data.iter().zip(&helper.data).enumerate() {
                assert!(
                    (t - h).abs() <= 1e-4 + 1e-4 * t.abs(),
                    "{label} mismatch at idx {i}: autograd = {t}, helper = {h}"
                );
            }
        };
        cmp("grad_x", &grad_x_truth, &grads.grad_x);
        cmp("grad_w_gate", &grad_w_gate_truth, &grads.grad_w_gate);
        cmp("grad_w_up", &grad_w_up_truth, &grads.grad_w_up);
        cmp("grad_w_down", &grad_w_down_truth, &grads.grad_w_down);
    }

    /// Cross-backend test for chunk 4.5.b: same FFN backward chain on
    /// `Tensor` and `CudaTensor` agrees within FP tolerance. Tighter
    /// tolerance than the chunk-4.5.a attention test because the FFN
    /// chain is shorter (3 matmuls + silu + mul vs 5 matmuls +
    /// softmax + 2 ropes + transpose) so less drift accumulates.
    #[cfg(feature = "cuda")]
    #[test]
    fn ffn_backward_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Realistic-ish v0.13 FFN shapes (seq scaled down for test
        // speed): seq=8, hidden=24, ffn=48 (2x ratio matching v0.13).
        let seq = 8usize;
        let hidden = 24usize;
        let ffn = 48usize;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.041).sin() * 0.6)
            .collect();
        let w_gate_data: Vec<f32> = (0..hidden * ffn)
            .map(|i| (i as f32 * 0.097).cos() * 0.3)
            .collect();
        let w_up_data: Vec<f32> = (0..hidden * ffn)
            .map(|i| (i as f32 * 0.131).sin() * 0.3)
            .collect();
        let w_down_data: Vec<f32> = (0..ffn * hidden)
            .map(|i| (i as f32 * 0.169).cos() * 0.3)
            .collect();
        let g_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.061).sin() + 0.1)
            .collect();

        let x_cpu = Tensor::from_vec(x_data.clone(), vec![seq, hidden]);
        let w_gate_cpu = Tensor::from_vec(w_gate_data.clone(), vec![hidden, ffn]);
        let w_up_cpu = Tensor::from_vec(w_up_data.clone(), vec![hidden, ffn]);
        let w_down_cpu = Tensor::from_vec(w_down_data.clone(), vec![ffn, hidden]);
        let g_cpu = Tensor::from_vec(g_data.clone(), vec![seq, hidden]);

        let (out_cpu, saved_cpu) = ffn_forward_save(&x_cpu, &w_gate_cpu, &w_up_cpu, &w_down_cpu);
        let grads_cpu = ffn_backward(
            &g_cpu,
            &x_cpu,
            &saved_cpu,
            &w_gate_cpu,
            &w_up_cpu,
            &w_down_cpu,
        );

        let x_gpu = CudaTensor::from_cpu(&x_cpu).expect("H->D x");
        let w_gate_gpu = CudaTensor::from_cpu(&w_gate_cpu).expect("H->D w_gate");
        let w_up_gpu = CudaTensor::from_cpu(&w_up_cpu).expect("H->D w_up");
        let w_down_gpu = CudaTensor::from_cpu(&w_down_cpu).expect("H->D w_down");
        let g_gpu = CudaTensor::from_cpu(&g_cpu).expect("H->D g");

        let (out_gpu, saved_gpu) = ffn_forward_save(&x_gpu, &w_gate_gpu, &w_up_gpu, &w_down_gpu);
        let grads_gpu = ffn_backward(
            &g_gpu,
            &x_gpu,
            &saved_gpu,
            &w_gate_gpu,
            &w_up_gpu,
            &w_down_gpu,
        );

        let assert_close = |label: &str, cpu_t: &Tensor, gpu_t_dev: &CudaTensor, tol: f32| {
            let gpu_t = gpu_t_dev.to_cpu().expect("D->H");
            assert_eq!(cpu_t.shape, gpu_t.shape, "{label}: shape diverged");
            for (i, (&c, &g)) in cpu_t.data.iter().zip(&gpu_t.data).enumerate() {
                let abs = (c - g).abs();
                assert!(
                    abs <= tol + tol * c.abs(),
                    "{label} drift at idx {i}: cpu = {c}, cuda = {g}"
                );
            }
        };
        assert_close("forward output", &out_cpu, &out_gpu, 1e-3);
        assert_close("grad_x", &grads_cpu.grad_x, &grads_gpu.grad_x, 5e-3);
        assert_close(
            "grad_w_gate",
            &grads_cpu.grad_w_gate,
            &grads_gpu.grad_w_gate,
            5e-3,
        );
        assert_close(
            "grad_w_up",
            &grads_cpu.grad_w_up,
            &grads_gpu.grad_w_up,
            5e-3,
        );
        assert_close(
            "grad_w_down",
            &grads_cpu.grad_w_down,
            &grads_gpu.grad_w_down,
            5e-3,
        );
    }

    // ---- Phase 4 chunk 4.5.c: full transformer block forward+backward ----

    /// Helper for chunk-4.5.c tests: build a small `HeadWeights<Tensor>`
    /// and `FfnWeights<Tensor>` set with deterministic non-trivial values.
    /// Shapes match the standard sum-of-projections invariant
    /// `n_heads * head_dim == hidden`.
    #[allow(dead_code)]
    fn make_block_weights_cpu(
        hidden: usize,
        head_dim: usize,
        n_heads: usize,
        ffn: usize,
    ) -> (Vec<HeadWeights<Tensor>>, FfnWeights<Tensor>) {
        let mk = |rows: usize, cols: usize, seed: f32| {
            Tensor::from_vec(
                (0..rows * cols)
                    .map(|i| (i as f32 * seed).sin() * 0.3)
                    .collect(),
                vec![rows, cols],
            )
        };
        let mut heads = Vec::with_capacity(n_heads);
        for h in 0..n_heads {
            let off = (h as f32 + 1.0) * 0.013;
            heads.push(HeadWeights {
                w_q: mk(hidden, head_dim, 0.061 + off),
                w_k: mk(hidden, head_dim, 0.097 + off),
                w_v: mk(hidden, head_dim, 0.131 + off),
                w_o: mk(head_dim, hidden, 0.169 + off),
            });
        }
        let ffn = FfnWeights {
            w_gate: mk(hidden, ffn, 0.211),
            w_up: mk(hidden, ffn, 0.241),
            w_down: mk(ffn, hidden, 0.277),
        };
        (heads, ffn)
    }

    /// Forward output of `block_forward_save` must equal
    /// `block_inference` on the same inputs - the only difference
    /// between the two is whether the intermediates are owned in a
    /// returned struct or dropped at end-of-expression. Bit-equality
    /// is the right guarantee here since the math runs in the same
    /// order on the same backend.
    #[test]
    fn block_forward_save_output_equals_block_inference() {
        let hidden = 8usize;
        let head_dim = 4usize;
        let n_heads = 2usize;
        let ffn_dim = 16usize;
        let seq = 5usize;
        let x = Tensor::from_vec(
            (0..seq * hidden)
                .map(|i| (i as f32 * 0.073).sin() * 0.6)
                .collect(),
            vec![seq, hidden],
        );
        let (heads, ffn) = make_block_weights_cpu(hidden, head_dim, n_heads, ffn_dim);
        let out_inference = block_inference(&x, &heads, &ffn, head_dim);
        let (out_save, _saved) = block_forward_save(&x, &heads, &ffn, head_dim);
        assert_eq!(out_inference.shape, out_save.shape);
        for (i, (&a, &b)) in out_inference.data.iter().zip(&out_save.data).enumerate() {
            assert_eq!(
                a, b,
                "block_forward_save diverged from block_inference at idx {i}: \
                 inference = {a}, save = {b}"
            );
        }
    }

    /// Cross-backend agreement on a full transformer block: same
    /// `block_forward_save` + `block_backward` chain on `Tensor` and
    /// `CudaTensor` produces matching outputs and gradients within FP
    /// tolerance. **First test that exercises every Phase 4 backward
    /// kernel through a full block-shape graph on the GPU.** Together
    /// with the chunk-4.5.a / 4.5.b autograd-ground-truth tests this
    /// transitively gates correctness: those gate per-helper math
    /// against autograd; this gates that the same chain runs the same
    /// way on both backends.
    #[cfg(feature = "cuda")]
    #[test]
    fn block_backward_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Realistic-ish small block: hidden=16, head_dim=4 (so 4
        // heads), ffn=32 (2x ratio matching v0.13), seq=8.
        let hidden = 16usize;
        let head_dim = 4usize;
        let n_heads = 4usize;
        let ffn_dim = 32usize;
        let seq = 8usize;

        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.041).sin() * 0.5)
            .collect();
        let g_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.061).cos() + 0.1)
            .collect();
        let x_cpu = Tensor::from_vec(x_data.clone(), vec![seq, hidden]);
        let g_cpu = Tensor::from_vec(g_data.clone(), vec![seq, hidden]);
        let (heads_cpu, ffn_cpu) = make_block_weights_cpu(hidden, head_dim, n_heads, ffn_dim);

        let copy_to_gpu = |t: &Tensor| CudaTensor::from_cpu(t).expect("H->D");
        let heads_gpu: Vec<HeadWeights<CudaTensor>> = heads_cpu
            .iter()
            .map(|h| HeadWeights {
                w_q: copy_to_gpu(&h.w_q),
                w_k: copy_to_gpu(&h.w_k),
                w_v: copy_to_gpu(&h.w_v),
                w_o: copy_to_gpu(&h.w_o),
            })
            .collect();
        let ffn_gpu: FfnWeights<CudaTensor> = FfnWeights {
            w_gate: copy_to_gpu(&ffn_cpu.w_gate),
            w_up: copy_to_gpu(&ffn_cpu.w_up),
            w_down: copy_to_gpu(&ffn_cpu.w_down),
        };
        let x_gpu = copy_to_gpu(&x_cpu);
        let g_gpu = copy_to_gpu(&g_cpu);

        let (out_cpu, saved_cpu) = block_forward_save(&x_cpu, &heads_cpu, &ffn_cpu, head_dim);
        let grads_cpu = block_backward(&g_cpu, &x_cpu, &saved_cpu, &heads_cpu, &ffn_cpu, head_dim);
        let (out_gpu, saved_gpu) = block_forward_save(&x_gpu, &heads_gpu, &ffn_gpu, head_dim);
        let grads_gpu = block_backward(&g_gpu, &x_gpu, &saved_gpu, &heads_gpu, &ffn_gpu, head_dim);

        let assert_close = |label: &str, cpu_t: &Tensor, gpu_t_dev: &CudaTensor, tol: f32| {
            let gpu_t = gpu_t_dev.to_cpu().expect("D->H");
            assert_eq!(cpu_t.shape, gpu_t.shape, "{label}: shape diverged");
            for (i, (&c, &g)) in cpu_t.data.iter().zip(&gpu_t.data).enumerate() {
                let abs = (c - g).abs();
                assert!(
                    abs <= tol + tol * c.abs(),
                    "{label} drift at idx {i}: cpu = {c}, cuda = {g}"
                );
            }
        };
        // Block-level tolerances loosen further than chunk 4.5.a's
        // because the chain is now: rmsnorm + (attention head x N_heads
        // each chained through 5 matmuls + softmax + 2 ropes + transpose
        // + 3-branch sum) + add + rmsnorm + (3 matmuls + silu + mul) +
        // add. Drift accumulates multiplicatively.
        assert_close("forward output", &out_cpu, &out_gpu, 1e-2);
        assert_close("grad_x", &grads_cpu.grad_x, &grads_gpu.grad_x, 2e-2);
        assert_eq!(grads_cpu.head_grads.len(), grads_gpu.head_grads.len());
        for (idx, (cg, gg)) in grads_cpu
            .head_grads
            .iter()
            .zip(&grads_gpu.head_grads)
            .enumerate()
        {
            assert_close(
                &format!("head[{idx}].grad_w_q"),
                &cg.grad_w_q,
                &gg.grad_w_q,
                2e-2,
            );
            assert_close(
                &format!("head[{idx}].grad_w_k"),
                &cg.grad_w_k,
                &gg.grad_w_k,
                2e-2,
            );
            assert_close(
                &format!("head[{idx}].grad_w_v"),
                &cg.grad_w_v,
                &gg.grad_w_v,
                2e-2,
            );
            assert_close(
                &format!("head[{idx}].grad_w_o"),
                &cg.grad_w_o,
                &gg.grad_w_o,
                2e-2,
            );
        }
        assert_close(
            "grad_w_gate",
            &grads_cpu.grad_w_gate,
            &grads_gpu.grad_w_gate,
            2e-2,
        );
        assert_close(
            "grad_w_up",
            &grads_cpu.grad_w_up,
            &grads_gpu.grad_w_up,
            2e-2,
        );
        assert_close(
            "grad_w_down",
            &grads_cpu.grad_w_down,
            &grads_gpu.grad_w_down,
            2e-2,
        );
    }

    // ---- Phase 4 chunk 4.5.d: cross-entropy forward + backward ----

    /// Cross-entropy loss + backward must match `Var::cross_entropy`
    /// (autograd.rs:1025-1115) byte-for-byte: same subtract-max
    /// log-sum-exp trick, same `(softmax - onehot) / seq` gradient.
    /// Path A is autograd; path B is the new chunk-4.5.d helpers.
    #[test]
    fn cross_entropy_matches_autograd_ground_truth() {
        use crate::autograd::{Tape, Var};
        let seq = 6usize;
        let vocab = 11usize;
        let logits_data: Vec<f32> = (0..seq * vocab)
            .map(|i| (i as f32 * 0.083).sin() * 1.5)
            .collect();
        let targets: Vec<usize> = (0..seq).map(|i| (i * 3 + 1) % vocab).collect();

        // Path A: autograd.
        let tape = Tape::new();
        let logits_var = Var::leaf(
            &tape,
            Tensor::from_vec(logits_data.clone(), vec![seq, vocab]),
        );
        let loss_var = logits_var.cross_entropy(&targets);
        tape.backward(loss_var.id);
        let loss_truth = loss_var.value().data[0];
        let grad_logits_truth = logits_var.grad();

        // Path B: chunk-4.5.d helpers.
        let logits = Tensor::from_vec(logits_data, vec![seq, vocab]);
        let (loss_helper, softmax_saved) = cross_entropy_forward_save(&logits, &targets);
        let grad_logits_helper = cross_entropy_backward(&softmax_saved, &targets, seq);

        assert!(
            (loss_helper - loss_truth).abs() <= 1e-5 + 1e-5 * loss_truth.abs(),
            "loss mismatch: helper = {loss_helper}, autograd = {loss_truth}"
        );
        assert_eq!(grad_logits_helper.shape, grad_logits_truth.shape);
        for (i, (&h, &t)) in grad_logits_helper
            .data
            .iter()
            .zip(&grad_logits_truth.data)
            .enumerate()
        {
            assert!(
                (h - t).abs() <= 1e-5 + 1e-5 * t.abs(),
                "grad mismatch at idx {i}: helper = {h}, autograd = {t}"
            );
        }
    }

    /// Cross-backend agreement for the new fused cross-entropy
    /// kernels: same logits + same targets through CPU and CUDA paths
    /// produce matching loss scalars and matching gradient tensors.
    /// First test that exercises `cross_entropy_softmax_loss_f32` and
    /// `cross_entropy_backward_f32` on the GPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn cross_entropy_cpu_and_cuda_agree_within_tolerance() {
        use crate::cuda::{CudaTensor, cuda_state};
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Realistic-ish vocab + seq sizes.
        let seq = 13usize;
        let vocab = 67usize;
        let logits_data: Vec<f32> = (0..seq * vocab)
            .map(|i| (i as f32 * 0.041).sin() * 1.2)
            .collect();
        let targets: Vec<usize> = (0..seq).map(|i| (i * 7 + 3) % vocab).collect();
        let logits_cpu = Tensor::from_vec(logits_data.clone(), vec![seq, vocab]);
        let logits_gpu = CudaTensor::from_cpu(&logits_cpu).expect("H->D");
        let (loss_cpu, sm_cpu) = cross_entropy_forward_save(&logits_cpu, &targets);
        let (loss_gpu, sm_gpu) = cross_entropy_forward_save(&logits_gpu, &targets);
        let grad_cpu = cross_entropy_backward(&sm_cpu, &targets, seq);
        let grad_gpu = cross_entropy_backward(&sm_gpu, &targets, seq);
        let sm_gpu_host = sm_gpu.to_cpu().expect("D->H softmax");
        let grad_gpu_host = grad_gpu.to_cpu().expect("D->H grad_logits");

        assert!(
            (loss_cpu - loss_gpu).abs() <= 1e-4 + 1e-4 * loss_cpu.abs(),
            "loss mismatch: cpu = {loss_cpu}, cuda = {loss_gpu}"
        );
        assert_eq!(sm_cpu.shape, sm_gpu_host.shape);
        for (i, (&c, &g)) in sm_cpu.data.iter().zip(&sm_gpu_host.data).enumerate() {
            assert!(
                (c - g).abs() <= 1e-4 + 1e-4 * c.abs(),
                "softmax drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
        for (i, (&c, &g)) in grad_cpu.data.iter().zip(&grad_gpu_host.data).enumerate() {
            assert!(
                (c - g).abs() <= 1e-4 + 1e-4 * c.abs(),
                "grad drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }
}

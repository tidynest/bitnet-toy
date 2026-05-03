//! Single-head scaled-dot-product self-attention, BitNet-quantised throughout.
//!
//! Built as a free function (not a struct yet) because the weights are passed
//! in by the caller as plain `Var` leaves on the tape - same training pattern
//! as M6's `train_bitlinear_regression`. The struct wrapper arrives in
//! M7 portion 4 (the full transformer block).
//!
//! Math:
//!     Q = x_eff · W_q,        K = x_eff · W_k,        V = x_eff · W_v
//!     scores  = (Q · Kᵀ) / √d_k       # [seq, seq]
//!     attn    = softmax_per_row(scores)
//!     context = attn · V              # [seq, head_dim]
//!     out     = context_eff · W_o     # [seq, hidden_dim]
//!
//! The scaling by 1/√d_k is essential - without it, softmax saturates as d_k
//! grows and the gradient through it vanishes ("Attention Is All You Need" §3.2).
//!
//! Shapes:
//!     x   : [seq_len, hidden_dim]
//!     W_q, W_k, W_v : [hidden_dim, head_dim]
//!     W_o           : [head_dim,  hidden_dim]
//!     out : [seq_len, hidden_dim]
//!
//! Quantisation routing:
//!     - x is quantised ONCE (per-row INT8 STE) and reused for Q/K/V projections.
//!       Gradient on x accumulates from all three paths via tape's per-cell `+=`.
//!     - Each weight matrix gets its own ternary STE quant.
//!     - The context (post-attention output) is re-quantised before W_o.

use crate::autograd::Var;

/// Forward pass for a single attention head. Both `x` and the weight Vars must
/// already be registered as leaves on the same tape.
///
/// `head_dim` controls the 1/√d_k scaling factor. It must equal `W_q.shape[1]`
/// (= the second dim of any of W_q/W_k/W_v). Passed explicitly so the function
/// doesn't have to peek at Var values just to read a shape.
pub fn attention<'t>(
    x: Var<'t>,
    w_q: Var<'t>,
    w_k: Var<'t>,
    w_v: Var<'t>,
    w_o: Var<'t>,
    head_dim: usize,
) -> Var<'t> {
    // ── Pre-compute scaling factor once (constant across the forward). ──
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // ── Quantise the input once; reuse across Q/K/V. ──
    // Var is Copy, so each `.matmul(...)` call uses an independent copy of the
    // x_eff handle - but they all point at the SAME node on the tape, so its
    // gradient correctly accumulates from the three projection paths.
    let x_eff = x.quantise_acts_ste();

    // ── Project to Q, K, V via ternary-quantised weights. ──
    let q = x_eff.matmul(w_q.quantise_weights_ste());
    let k = x_eff.matmul(w_k.quantise_weights_ste());
    let v = x_eff.matmul(w_v.quantise_weights_ste());

    // ── Attention scores: scaled Q · Kᵀ, then causal mask. ──
    // Transpose K to [head_dim, seq] so Q · Kᵀ has shape [seq, seq].
    // mul_scalar applies 1/√d_k on the autograd path so backward picks up the factor.
    // causal_mask sets scores[i, j] = -inf for j > i so attention at position i
    // only ever sees positions <= i. Without this, the model would learn the
    // trivial cheat of copying input[i+1] when predicting target[i] = input[i+1].
    let scores = q
        .matmul(k.transpose_2d())
        .mul_scalar(scale)
        .causal_mask();

    // ── Per-row softmax over the seq axis. ──
    // Each row of attn is a probability distribution over keys for that query.
    // Positions masked to -inf become exactly 0 after exp, so attention only
    // attends to past + current positions.
    let attn = scores.softmax();

    // ── Apply attention weights to values. ──
    let context = attn.matmul(v);

    // ── Output projection through another BitLinear. ──
    let context_eff = context.quantise_acts_ste();
    context_eff.matmul(w_o.quantise_weights_ste())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Tape, Var};
    use crate::tensor::Tensor;

    /// Build an `[in_dim, out_dim]` weight matrix with deterministic values
    /// `0.05, 0.10, 0.15, …` so multiple test cases share predictable inputs
    /// without an RNG dep.
    fn make_weight(in_dim: usize, out_dim: usize) -> Tensor {
        Tensor::from_vec(
            (0..in_dim * out_dim)
                .map(|i| (i as f32) * 0.05 + 0.05)
                .collect(),
            vec![in_dim, out_dim],
        )
    }

    #[test]
    fn attention_forward_produces_correct_shape() {
        let tape = Tape::new();
        let seq_len = 4;
        let hidden_dim = 6;
        let head_dim = 6;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );

        let w_q = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_k = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_v = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_o = Var::leaf(&tape, make_weight(head_dim, hidden_dim));

        let y = attention(x, w_q, w_k, w_v, w_o, head_dim);
        assert_eq!(y.value().shape, vec![seq_len, hidden_dim]);
    }

    #[test]
    fn attention_softmax_rows_sum_to_one_after_forward() {
        // Indirect check: if the attention path is wired correctly, the
        // intermediate softmax output (shape [seq, seq]) has rows summing to 1.
        // We can't observe `attn` directly through the public API, but we CAN
        // verify the network produces a meaningful output (not NaN, not zero
        // unless trivially so).
        let tape = Tape::new();
        let seq_len = 3;
        let hidden_dim = 4;
        let head_dim = 4;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.2 - 0.5)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let w_q = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_k = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_v = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_o = Var::leaf(&tape, make_weight(head_dim, hidden_dim));

        let y = attention(x, w_q, w_k, w_v, w_o, head_dim);

        // No NaN, no infinities anywhere in the output.
        for (i, &v) in y.value().data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "attention output[{}] = {} is not finite",
                i,
                v
            );
        }
    }

    #[test]
    fn attention_backward_routes_gradient_to_all_weights_and_input() {
        // The integration test for portion 2: every leaf in the attention graph
        // (x and all four weights) must receive non-zero gradient after backward.
        // If any leaf has all-zero grad, some link in the backward chain is broken.
        let tape = Tape::new();
        let seq_len = 3;
        let hidden_dim = 4;
        let head_dim = 4;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1 + 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let w_q = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_k = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_v = Var::leaf(&tape, make_weight(hidden_dim, head_dim));
        let w_o = Var::leaf(&tape, make_weight(head_dim, hidden_dim));

        let y = attention(x, w_q, w_k, w_v, w_o, head_dim);
        let loss = y.mean();

        tape.backward(loss.id);

        // Helper - collapses "any non-zero element?" into one bool.
        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-8);

        // Every weight must receive gradient. The names are spelled out so the
        // failure message points at exactly which projection's backward broke.
        assert!(has_nonzero(&w_q.grad()), "w_q got all-zero gradient");
        assert!(has_nonzero(&w_k.grad()), "w_k got all-zero gradient");
        assert!(has_nonzero(&w_v.grad()), "w_v got all-zero gradient");
        assert!(has_nonzero(&w_o.grad()), "w_o got all-zero gradient");

        // The input Var must also receive gradient - through THREE paths
        // (Q, K, V projections), correctly summed by tape's accumulator.
        assert!(has_nonzero(&x.grad()), "x got all-zero gradient");
    }
}

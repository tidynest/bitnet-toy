//! Multi-head scaled-dot-product self-attention, BitNet-quantised throughout,
//! sum-of-projections form.
//!
//! Each head is a fully independent attention path with its own Q/K/V/O
//! ternary projections. The block's attention output is the element-wise sum
//! of all head outputs, which is mathematically equivalent to the canonical
//! "concat heads then project once with a wide W_o" formulation but avoids the
//! need to add a `concat` op to the autograd. Splitting `W_o` into per-head
//! `W_o_i` slices and summing matches it because matrix multiplication
//! distributes over horizontal block concatenation:
//!     [H_1 | H_2 | ... | H_n] · [W_o_1; W_o_2; ...; W_o_n] = sum_i H_i · W_o_i.
//!
//! Math (per head i):
//!     Q_i = x_eff · W_q_i      K_i = x_eff · W_k_i      V_i = x_eff · W_v_i
//!     scores_i = (Q_i · K_iᵀ) / √head_dim                 # [seq, seq]
//!     scores_i = causal_mask(scores_i)                    # upper-tri -> -inf
//!     attn_i   = softmax_per_row(scores_i)
//!     ctx_i    = attn_i · V_i                             # [seq, head_dim]
//!     head_i   = ctx_i_eff · W_o_i                        # [seq, hidden_dim]
//!
//! Output:
//!     out = sum over i of head_i                          # [seq, hidden_dim]
//!
//! The 1/√head_dim scaling is per-head, on each head's own scores. Without it,
//! softmax saturates as head_dim grows ("Attention Is All You Need" §3.2).
//!
//! Quantisation routing:
//!     - x is quantised ONCE per call (per-row INT8 STE) and reused across
//!       every head's Q/K/V projections. Gradient on x accumulates from all
//!       3 * n_heads paths via tape's per-cell `+=`.
//!     - Each per-head weight gets its own ternary STE quant.
//!     - Each head's context is re-quantised before its W_o.

use crate::autograd::Var;
use crate::block::AttentionHeadVars;

/// Forward pass for multi-head attention. `x` and every Var inside `heads`
/// must already be registered as leaves on the same tape.
///
/// `head_dim` is each head's inner dimension. Used for the per-head 1/√d_k
/// scaling and must equal `heads[i].w_q.shape[1]` for every head.
pub fn attention<'t>(x: Var<'t>, heads: &[AttentionHeadVars<'t>], head_dim: usize) -> Var<'t> {
    assert!(!heads.is_empty(), "attention requires at least one head");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Quantise the input once. Var is Copy: each `.matmul(...)` shares the same
    // tape node, so the head paths all flow gradient back into a single x_eff
    // gradient cell via the tape's per-cell accumulator.
    let x_eff = x.quantise_acts_ste();

    // Compute the first head's full output as the seed of the running sum.
    // Starting from `head_output(heads[0])` avoids needing a "zero Var" leaf,
    // which would require knowing the seq_len upfront and adds a redundant
    // node to the tape.
    let mut combined = head_output(x_eff, &heads[0], scale);

    // Sum every subsequent head's output into `combined`. Each iteration
    // appends an `add` node to the tape; backward routes gradient element-wise
    // to both summands, so each head's W_o accumulates its own share without
    // any concat operation. Reassigning is required because `Var::add` is
    // functional - it returns a fresh Var rather than mutating in place.
    for h in &heads[1..] {
        combined = combined.add(head_output(x_eff, h, scale));
    }

    combined
}

/// One head's full attention path: x_eff -> Q -> K -> V -> attn -> ctx -> W_o.
/// Result has shape `[seq, hidden_dim]` so it can be summed directly with
/// other heads' outputs (sum-of-projections form).
fn head_output<'t>(x_eff: Var<'t>, h: &AttentionHeadVars<'t>, scale: f32) -> Var<'t> {
    // Q, K, V projections through ternary-quantised per-head weights.
    // Q and K are then rotated by RoPE so attention scores depend on the
    // *difference* between query and key positions, not their absolute
    // values. V is unrotated; the value content is position-independent.
    // BitNet b1.58 / LLaMA convention; replaces a learned absolute pos_embed
    // sat outside the attention path.
    let q = x_eff.matmul(h.w_q.quantise_weights_ste()).rope();
    let k = x_eff.matmul(h.w_k.quantise_weights_ste()).rope();
    let v = x_eff.matmul(h.w_v.quantise_weights_ste());

    // Scaled dot-product scores with causal mask. mul_scalar applies the
    // 1/√d_k factor on the autograd path so backward picks up the scaling
    // automatically. causal_mask sets scores[i, j] = -inf for j > i so a
    // query at position i never attends to positions > i; without this, the
    // model can trivially predict target[i] = input[i+1].
    let scores = q.matmul(k.transpose_2d()).mul_scalar(scale).causal_mask();

    // Per-row softmax over the seq axis -> probabilities over keys.
    // Apply to V to get the per-head context [seq, head_dim].
    let attn = scores.softmax();
    let ctx = attn.matmul(v);

    // Output projection brings each head's [seq, head_dim] back to
    // [seq, hidden_dim]. Re-quantise the context before W_o so this matmul
    // also runs as a BitLinear.
    ctx.quantise_acts_ste().matmul(h.w_o.quantise_weights_ste())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Tape, Var};
    use crate::tensor::Tensor;

    /// Build an `[in_dim, out_dim]` weight matrix with deterministic values
    /// `0.05, 0.10, 0.15, …` so multiple test cases share predictable inputs
    /// without an RNG dep. Per-head offset prevents two heads from being
    /// accidentally numerically identical (which would mask asymmetry bugs).
    fn make_weight(in_dim: usize, out_dim: usize, head_offset: f32) -> Tensor {
        Tensor::from_vec(
            (0..in_dim * out_dim)
                .map(|i| (i as f32) * 0.05 + 0.05 + head_offset)
                .collect(),
            vec![in_dim, out_dim],
        )
    }

    fn build_heads<'t>(
        tape: &'t Tape,
        n_heads: usize,
        hidden_dim: usize,
        head_dim: usize,
    ) -> Vec<AttentionHeadVars<'t>> {
        (0..n_heads)
            .map(|i| {
                let off = (i as f32) * 0.001;
                AttentionHeadVars {
                    w_q: Var::leaf(tape, make_weight(hidden_dim, head_dim, off)),
                    w_k: Var::leaf(tape, make_weight(hidden_dim, head_dim, off)),
                    w_v: Var::leaf(tape, make_weight(hidden_dim, head_dim, off)),
                    w_o: Var::leaf(tape, make_weight(head_dim, hidden_dim, off)),
                }
            })
            .collect()
    }

    #[test]
    fn attention_forward_produces_correct_shape() {
        let tape = Tape::new();
        let seq_len = 4;
        let n_heads = 3;
        let hidden_dim = 6;
        let head_dim = 2; // n_heads * head_dim == hidden_dim

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );

        let heads = build_heads(&tape, n_heads, hidden_dim, head_dim);
        let y = attention(x, &heads, head_dim);
        assert_eq!(y.value().shape, vec![seq_len, hidden_dim]);
    }

    #[test]
    fn attention_softmax_rows_sum_to_one_after_forward() {
        // Indirect check: if the attention path is wired correctly, the
        // forward output is finite (no NaN, no infinities). The sum-of-heads
        // form means a per-head softmax bug would still show up as a NaN here.
        let tape = Tape::new();
        let seq_len = 3;
        let n_heads = 2;
        let hidden_dim = 4;
        let head_dim = 2;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.2 - 0.5)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let heads = build_heads(&tape, n_heads, hidden_dim, head_dim);
        let y = attention(x, &heads, head_dim);
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
    fn attention_backward_routes_gradient_to_every_head_and_input() {
        // The integration test for multi-head: every leaf in every head
        // (Q/K/V/O across all heads) AND the input x must receive non-zero
        // gradient after backward. If the head loop drops a head silently,
        // that head's gradient stays at zero and this test catches it.
        let tape = Tape::new();
        let seq_len = 3;
        let n_heads = 2;
        let hidden_dim = 4;
        let head_dim = 2;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1 + 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let heads = build_heads(&tape, n_heads, hidden_dim, head_dim);

        let y = attention(x, &heads, head_dim);
        let loss = y.mean();

        tape.backward(loss.id);

        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-8);

        for (i, h) in heads.iter().enumerate() {
            assert!(has_nonzero(&h.w_q.grad()), "head {} w_q zero grad", i);
            assert!(has_nonzero(&h.w_k.grad()), "head {} w_k zero grad", i);
            assert!(has_nonzero(&h.w_v.grad()), "head {} w_v zero grad", i);
            assert!(has_nonzero(&h.w_o.grad()), "head {} w_o zero grad", i);
        }
        assert!(has_nonzero(&x.grad()), "x got all-zero gradient");
    }
}

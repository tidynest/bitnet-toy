//! Pre-norm BitNet transformer block.
//!
//! Forward (LLaMA / BitNet style):
//!     x1 = x  + attention(rmsnorm(x))
//!     x2 = x1 + ffn(rmsnorm(x1))
//!     return x2
//!
//! Pre-norm (RMSNorm before each sublayer, not after) is the modern default -
//! more stable in deep stacks. Residual connections give gradients an unobstructed
//! path back to early layers, which is what makes deep transformers trainable
//! at all.
//!
//! Six weight tensors per block: Q/K/V/O for attention, up/down for FFN.
//! RMSNorm has no learnable gain in this implementation, so no norm weights.

use crate::attention::attention;
use crate::autograd::Var;
use crate::ffn::ffn;

/// Bundle of per-block weight Vars.  All must live on the same tape as the
/// input `x` passed to `transformer_block`. The struct is purely an argument-
/// packaging convenience - six positional Vars would be cumbersome and easy
/// to mis-order at the call site.
pub struct BlockWeights<'t> {
    pub attn_w_q: Var<'t>,
    pub attn_w_k: Var<'t>,
    pub attn_w_v: Var<'t>,
    pub attn_w_o: Var<'t>,
    pub ffn_up_w: Var<'t>,
    pub ffn_down_w: Var<'t>,
}

/// Forward through one transformer block.
///
/// Shapes:
///     x : [seq_len, hidden_dim]
///     out : [seq_len, hidden_dim]
///
/// `head_dim` is the size of the (single) attention head - used for the
/// `1/√d_k` scaling inside `attention`. Must match the second dim of W_q/W_k/W_v.
pub fn transformer_block<'t>(x: Var<'t>, w: &BlockWeights<'t>, head_dim: usize) -> Var<'t> {
    // ── Sublayer 1: pre-norm + attention + residual. ──
    // The residual `x.add(...)` adds the *unmodified* input back to the
    // sublayer's output - this is the gradient highway that makes deep
    // transformers trainable.
    let attn_out = attention(
        x.rmsnorm(),
        w.attn_w_q,
        w.attn_w_k,
        w.attn_w_v,
        w.attn_w_o,
        head_dim,
    );
    let x1 = x.add(attn_out);

    // ── Sublayer 2: pre-norm + FFN + residual. ──
    let ffn_out = ffn(x1.rmsnorm(), w.ffn_up_w, w.ffn_down_w);
    x1.add(ffn_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Tape, Var};
    use crate::tensor::Tensor;

    fn make_weight(in_dim: usize, out_dim: usize, offset: f32) -> Tensor {
        // Deterministic + per-block offset so stacked-block tests don't accidentally
        // produce identical layers (which would mask asymmetry-related bugs).
        Tensor::from_vec(
            (0..in_dim * out_dim)
                .map(|i| (i as f32) * 0.05 + 0.05 + offset)
                .collect(),
            vec![in_dim, out_dim],
        )
    }

    fn make_block_weights<'t>(
        tape: &'t Tape,
        hidden_dim: usize,
        head_dim: usize,
        ffn_dim: usize,
        offset: f32,
    ) -> BlockWeights<'t> {
        BlockWeights {
            attn_w_q: Var::leaf(tape, make_weight(hidden_dim, head_dim, offset)),
            attn_w_k: Var::leaf(tape, make_weight(hidden_dim, head_dim, offset)),
            attn_w_v: Var::leaf(tape, make_weight(hidden_dim, head_dim, offset)),
            attn_w_o: Var::leaf(tape, make_weight(head_dim, hidden_dim, offset)),
            ffn_up_w: Var::leaf(tape, make_weight(hidden_dim, ffn_dim, offset)),
            ffn_down_w: Var::leaf(tape, make_weight(ffn_dim, hidden_dim, offset)),
        }
    }

    #[test]
    fn transformer_block_forward_preserves_shape() {
        // Fundamental shape contract: a block is shape-preserving.
        // [seq, hidden] in → [seq, hidden] out.  This is what makes blocks stackable.
        let tape = Tape::new();
        let seq_len = 4;
        let hidden_dim = 6;
        let head_dim = 6;
        let ffn_dim = 12;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1 + 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let w = make_block_weights(&tape, hidden_dim, head_dim, ffn_dim, 0.0);

        let y = transformer_block(x, &w, head_dim);
        assert_eq!(y.value().shape, vec![seq_len, hidden_dim]);

        for &v in y.value().data.iter() {
            assert!(
                v.is_finite(),
                "block output contains non-finite value: {}",
                v
            );
        }
    }

    #[test]
    fn transformer_block_backward_routes_gradient_everywhere() {
        // The integration test for the whole block: every weight in BOTH
        // sublayers, plus the input, must receive non-zero gradient.
        // Catches any broken-link bug across the ~30-node tape this builds.
        let tape = Tape::new();
        let seq_len = 3;
        let hidden_dim = 4;
        let head_dim = 4;
        let ffn_dim = 8;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1 + 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let w = make_block_weights(&tape, hidden_dim, head_dim, ffn_dim, 0.0);

        let y = transformer_block(x, &w, head_dim);
        let loss = y.mean();
        tape.backward(loss.id);

        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-8);

        assert!(
            has_nonzero(&w.attn_w_q.grad()),
            "attn_w_q got all-zero gradient"
        );
        assert!(
            has_nonzero(&w.attn_w_k.grad()),
            "attn_w_k got all-zero gradient"
        );
        assert!(
            has_nonzero(&w.attn_w_v.grad()),
            "attn_w_v got all-zero gradient"
        );
        assert!(
            has_nonzero(&w.attn_w_o.grad()),
            "attn_w_o got all-zero gradient"
        );
        assert!(
            has_nonzero(&w.ffn_up_w.grad()),
            "ffn_up_w got all-zero gradient"
        );
        assert!(
            has_nonzero(&w.ffn_down_w.grad()),
            "ffn_down_w got all-zero gradient"
        );
        assert!(has_nonzero(&x.grad()), "x got all-zero gradient");
    }

    #[test]
    fn stacked_two_blocks_forward_and_backward() {
        // The real M7 gate: two blocks chained back-to-back, all 12 weight
        // matrices participating in one forward + backward pass.
        // If this works, the architecture is structurally trainable; M9's
        // training loop just needs to add the LM head and the data pipeline.
        let tape = Tape::new();
        let seq_len = 3;
        let hidden_dim = 4;
        let head_dim = 4;
        let ffn_dim = 8;

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1 + 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );

        // Two blocks with slightly different weight offsets so they don't
        // accidentally produce identical sublayer outputs.
        let w0 = make_block_weights(&tape, hidden_dim, head_dim, ffn_dim, 0.00);
        let w1 = make_block_weights(&tape, hidden_dim, head_dim, ffn_dim, 0.01);

        let h = transformer_block(x, &w0, head_dim);
        let y = transformer_block(h, &w1, head_dim);
        let loss = y.mean();

        assert_eq!(y.value().shape, vec![seq_len, hidden_dim]);

        tape.backward(loss.id);

        // Every weight across BOTH blocks must receive gradient.
        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-8);
        for (name, var) in [
            ("w0.attn_w_q", w0.attn_w_q),
            ("w0.attn_w_k", w0.attn_w_k),
            ("w0.attn_w_v", w0.attn_w_v),
            ("w0.attn_w_o", w0.attn_w_o),
            ("w0.ffn_up_w", w0.ffn_up_w),
            ("w0.ffn_down_w", w0.ffn_down_w),
            ("w1.attn_w_q", w1.attn_w_q),
            ("w1.attn_w_k", w1.attn_w_k),
            ("w1.attn_w_v", w1.attn_w_v),
            ("w1.attn_w_o", w1.attn_w_o),
            ("w1.ffn_up_w", w1.ffn_up_w),
            ("w1.ffn_down_w", w1.ffn_down_w),
        ] {
            assert!(has_nonzero(&var.grad()), "{} got all-zero gradient", name);
        }
        // And the input.
        assert!(
            has_nonzero(&x.grad()),
            "x got all-zero gradient through stacked blocks"
        );
    }
}

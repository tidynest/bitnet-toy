//! Position-wise feed-forward network for the BitNet transformer block.
//!
//! Shape pipeline:
//!     x      : [seq_len, hidden_dim]
//!     up_w   : [hidden_dim, ffn_dim]      (typically ffn_dim ≈ 4 · hidden_dim)
//!     down_w : [ffn_dim,  hidden_dim]
//!     y      : [seq_len, hidden_dim]
//!
//! Forward:
//!     h = relu( quant(x)  · quant(up_w)   )      # [seq, ffn_dim]
//!     y =       quant(h)  · quant(down_w)        # [seq, hidden_dim]
//!
//! Same quant-routing pattern as attention: input quantised once per BitLinear,
//! intermediates re-quantised between projections. ReLU stands in for the
//! SwiGLU the BitNet paper actually uses - simpler (no gate weight, no extra
//! multiply), demonstrates the same position-wise mixing role.
//!
//! Authored as a free function for now; a `Ffn` struct may emerge in M7 portion 4
//! if the transformer-block wrapper benefits from grouping the two weights.

use crate::autograd::Var;

pub fn ffn<'t>(x: Var<'t>, up_w: Var<'t>, down_w: Var<'t>) -> Var<'t> {
    // Up-projection through BitLinear, then non-linearity.
    // Method-chain style makes the data flow read top-to-bottom: quant → matmul → relu.
    let h = x
        .quantise_acts_ste()
        .matmul(up_w.quantise_weights_ste())
        .relu();

    // Down-projection through a fresh BitLinear. Post-ReLU values get
    // re-quantised because their per-row magnitude distribution differs from
    // x's (rows where many features fired vs few).
    h.quantise_acts_ste().matmul(down_w.quantise_weights_ste())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Tape, Var};
    use crate::tensor::Tensor;

    /// Same deterministic-weight helper as `attention.rs`. Replicated here
    /// rather than shared because each test module is its own scope and the
    /// helper is six lines - the cost of a `pub(crate)` shared utility module
    /// would outweigh the duplication at this scale.
    fn make_weight(in_dim: usize, out_dim: usize) -> Tensor {
        Tensor::from_vec(
            (0..in_dim * out_dim)
                .map(|i| (i as f32) * 0.05 + 0.05)
                .collect(),
            vec![in_dim, out_dim],
        )
    }

    #[test]
    fn ffn_forward_produces_correct_shape() {
        let tape = Tape::new();
        let seq_len = 3;
        let hidden_dim = 4;
        let ffn_dim = 8; // 2× hidden_dim - keeps the test cheap; real models use 4×

        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq_len * hidden_dim)
                    .map(|i| (i as f32) * 0.1)
                    .collect(),
                vec![seq_len, hidden_dim],
            ),
        );
        let up_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim));
        let down_w = Var::leaf(&tape, make_weight(ffn_dim, hidden_dim));

        let y = ffn(x, up_w, down_w);
        assert_eq!(y.value().shape, vec![seq_len, hidden_dim]);

        // Cheap "did anything explode?" sanity - no NaN, no infinity.
        for &v in y.value().data.iter() {
            assert!(v.is_finite(), "FFN output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn ffn_backward_routes_gradient_to_all_weights_and_input() {
        // Same structural integration test as attention's: every leaf gets
        // a non-zero gradient. Catches "forward composes but backward chain
        // is broken at one link" bugs that pure-forward shape tests miss.
        let tape = Tape::new();
        let seq_len = 3;
        let hidden_dim = 4;
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
        let up_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim));
        let down_w = Var::leaf(&tape, make_weight(ffn_dim, hidden_dim));

        let y = ffn(x, up_w, down_w);
        let loss = y.mean();
        tape.backward(loss.id);

        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-8);
        assert!(has_nonzero(&up_w.grad()), "up_w got all-zero gradient");
        assert!(has_nonzero(&down_w.grad()), "down_w got all-zero gradient");
        assert!(has_nonzero(&x.grad()), "x got all-zero gradient");
    }

    #[test]
    fn ffn_forward_passes_through_relu_correctly() {
        // Sanity: with a *negative*-output up-projection, ReLU should kill
        // every intermediate, and the final output should therefore be zero.
        // Construct this by giving negative weights and positive input.
        let tape = Tape::new();
        let seq_len = 2;
        let hidden_dim = 3;
        let ffn_dim = 4;

        let x = Var::leaf(&tape, Tensor::ones(vec![seq_len, hidden_dim])); // all 1s

        // up_w = all -1: x · up_w = (sum of x) · (-1) = -3 per element. ReLU kills it.
        let neg_up = Tensor::from_vec(vec![-1.0; hidden_dim * ffn_dim], vec![hidden_dim, ffn_dim]);
        let up_w = Var::leaf(&tape, neg_up);
        let down_w = Var::leaf(&tape, make_weight(ffn_dim, hidden_dim));

        let y = ffn(x, up_w, down_w);

        // Every output element should be exactly zero (ReLU killed h, so
        // h_eff·down_w = 0, and the final BitLinear's quant of all-zeros
        // produces zeros via the α=0 row guard).
        assert!(
            y.value().data.iter().all(|&v| v == 0.0),
            "FFN output should be all zeros after ReLU killed h, got {:?}",
            y.value().data
        );
    }
}

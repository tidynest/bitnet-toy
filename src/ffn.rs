//! Position-wise SwiGLU feed-forward network for the BitNet transformer block.
//!
//! Shape pipeline:
//!     x       : [seq_len, hidden_dim]
//!     gate_w  : [hidden_dim, ffn_dim]   (typically ffn_dim ≈ 4 · hidden_dim)
//!     up_w    : [hidden_dim, ffn_dim]
//!     down_w  : [ffn_dim,   hidden_dim]
//!     y       : [seq_len,   hidden_dim]
//!
//! Forward (SwiGLU, the FFN form used by LLaMA / BitNet b1.58):
//!     gate = silu( quant(x) · quant(gate_w) )      # [seq, ffn_dim]
//!     up   =       quant(x) · quant(up_w)          # [seq, ffn_dim]
//!     h    = gate ⊙ up                              # [seq, ffn_dim]
//!     y    =        quant(h) · quant(down_w)       # [seq, hidden_dim]
//!
//! Why SwiGLU vs ReLU:
//!   - The element-wise product `gate ⊙ up` is a *gating* operation: the
//!     gate (post-SiLU) decides per-channel how much of the up-projection's
//!     value to pass through. SwiGLU lets the FFN learn to suppress
//!     irrelevant features at the per-position level rather than relying on
//!     the down-projection alone to do this work.
//!   - SiLU on the gate keeps the activation differentiable everywhere
//!     (no dead-neuron problem) and lets the gate value continuously
//!     interpolate between "fully open" (large positive activation),
//!     "closed" (near zero), and "slightly negative" (near zero with a
//!     tiny dip).
//!   - The up-projection is left linear so the gate alone determines
//!     non-linearity. Empirically SwiGLU outperforms ReLU and GELU at
//!     equal parameter count in language modelling.
//!
//! Quant routing: x is quantised once and reused for both the gate and up
//! projections (per-cell gradient accumulator handles the two paths). The
//! gated product `h` is re-quantised before the down projection.

use crate::autograd::Var;

pub fn ffn<'t>(x: Var<'t>, gate_w: Var<'t>, up_w: Var<'t>, down_w: Var<'t>) -> Var<'t> {
    // Quantise the input once; reuse for gate and up projections. Var is Copy,
    // so the two `.matmul` calls share the same x_eff tape node and the
    // tape's per-cell `+=` collects gradient from both paths back into the
    // single x_eff gradient cell.
    let x_eff = x.quantise_acts_ste();

    // Gate path: x · W_gate, then SiLU. SiLU is the non-linearity here; if
    // it is broken or returns identity, the gate becomes a plain linear
    // multiplier and SwiGLU collapses to a normal gated linear unit (still
    // trainable but loses the smooth gating behaviour).
    let gate = x_eff.matmul(gate_w.quantise_weights_ste()).silu();

    // Up path: plain BitLinear, no activation. The lack of non-linearity
    // here is intentional - the gate provides all the non-linearity in
    // SwiGLU.
    let up = x_eff.matmul(up_w.quantise_weights_ste());

    // Element-wise gated product. Var::mul broadcasts shape-equal tensors
    // element-wise; both `gate` and `up` are [seq, ffn_dim].
    let h = gate.mul(up);

    // Down-projection through a fresh BitLinear. Re-quantise the gated
    // product because its per-row magnitude distribution differs from x's.
    h.quantise_acts_ste().matmul(down_w.quantise_weights_ste())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Tape, Var};
    use crate::tensor::Tensor;

    /// Deterministic weight tensor with a per-call offset so the gate, up,
    /// and down weights aren't accidentally numerically identical (which
    /// would mask asymmetry-related bugs in the gating math).
    fn make_weight(in_dim: usize, out_dim: usize, offset: f32) -> Tensor {
        Tensor::from_vec(
            (0..in_dim * out_dim)
                .map(|i| (i as f32) * 0.05 + 0.05 + offset)
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
        let gate_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim, 0.00));
        let up_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim, 0.01));
        let down_w = Var::leaf(&tape, make_weight(ffn_dim, hidden_dim, 0.02));

        let y = ffn(x, gate_w, up_w, down_w);
        assert_eq!(y.value().shape, vec![seq_len, hidden_dim]);

        for &v in y.value().data.iter() {
            assert!(v.is_finite(), "FFN output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn ffn_backward_routes_gradient_to_all_three_weights_and_input() {
        // Every leaf must receive non-zero gradient: gate_w (through silu and
        // mul-with-up), up_w (through mul-with-gate), down_w (final matmul),
        // and x (through both gate and up paths via the shared x_eff).
        // If any path is broken, the corresponding leaf's gradient stays zero.
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
        let gate_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim, 0.00));
        let up_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim, 0.01));
        let down_w = Var::leaf(&tape, make_weight(ffn_dim, hidden_dim, 0.02));

        let y = ffn(x, gate_w, up_w, down_w);
        let loss = y.mean();
        tape.backward(loss.id);

        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-8);
        assert!(has_nonzero(&gate_w.grad()), "gate_w got all-zero gradient");
        assert!(has_nonzero(&up_w.grad()), "up_w got all-zero gradient");
        assert!(has_nonzero(&down_w.grad()), "down_w got all-zero gradient");
        assert!(has_nonzero(&x.grad()), "x got all-zero gradient");
    }

    #[test]
    fn ffn_silu_gate_keeps_negative_inputs_alive() {
        // Where ReLU would kill a strongly-negative pre-activation entirely,
        // SiLU lets a small negative-valued gate slip through, so the FFN
        // output stays non-zero. This is the qualitative difference between
        // ReLU-FFN and SwiGLU: SiLU is differentiable everywhere and avoids
        // dead neurons. Construct a deliberately negative gate path and
        // verify the output is finite but not all-zero.
        let tape = Tape::new();
        let seq_len = 2;
        let hidden_dim = 3;
        let ffn_dim = 4;

        let x = Var::leaf(&tape, Tensor::ones(vec![seq_len, hidden_dim]));

        // gate_w all -1: pre-silu gate values are very negative -> silu gives
        // small negative numbers near zero. Mul with up gives a small
        // non-zero gated product, and the down projection turns that into a
        // non-zero output (unlike ReLU which would have killed everything).
        let neg_gate = Tensor::from_vec(
            vec![-1.0; hidden_dim * ffn_dim],
            vec![hidden_dim, ffn_dim],
        );
        let gate_w = Var::leaf(&tape, neg_gate);
        let up_w = Var::leaf(&tape, make_weight(hidden_dim, ffn_dim, 0.01));
        let down_w = Var::leaf(&tape, make_weight(ffn_dim, hidden_dim, 0.02));

        let y = ffn(x, gate_w, up_w, down_w);

        for &v in y.value().data.iter() {
            assert!(v.is_finite(), "non-finite output through SiLU gate: {}", v);
        }
        // Output should not be exactly zero - SiLU's leakiness keeps the
        // gradient path alive even with a very negative gate pre-activation.
        // (A tiny non-zero floor is fine; the point is that something
        // survived where ReLU would have killed everything.)
        let any_nonzero = y.value().data.iter().any(|&v| v.abs() > 1e-12);
        assert!(
            any_nonzero,
            "SiLU gate produced exactly zero output; expected a small non-zero value"
        );
    }
}

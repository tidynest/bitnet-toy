//! BitLinear quantisers - ternary weights, INT8 per-token activations
//! (BitNet b1.58 §2).
//!
//! Two free functions:
//!   - `absmean_ternary` - quantise weights to {−1, 0, +1} with a scalar scale γ.
//!   - `absmax_int8`     - quantise activations per row to [−128, 127] with α per row.
//!
//! These are the only quantisation primitives the project needs. Higher-level
//! Var ops in `autograd.rs` (`quantise_weights_ste`, `quantise_acts_ste`) wrap
//! them with the straight-through-estimator backward.

use crate::tensor::Tensor;

/// Numerical-stability epsilon for the absmean-ternary divisor.
/// Prevents division by zero when a weight tensor is all zeros (degenerate but
/// possible during init, or for fully-pruned channels). Small enough to be inert
/// at any realistic weight magnitude.
const TERNARY_EPS: f32 = 1e-5;

/// Absmean-ternary weight quantisation (BitNet b1.58 §2).
///
/// Forward:
///     γ   = mean(|W|)                                     # single scalar
///     W_q = clamp(round(W / (γ + ε)), -1, +1)             # values in {-1, 0, +1}
///
/// Returned shapes:
///     W_q has the same shape as W; values are f32 but represent ternary integers.
///     γ   is a single f32 (the caller multiplies by it directly during rescale).
///
/// On dequantisation the effective weight is `γ · W_q`.
/// `γ` carries magnitude; `W_q` carries only sign + "is it zero?" - about
/// log₂(3) ≈ 1.58 bits of information per weight (the "b1.58" in the paper title).
pub fn absmean_ternary(w: &Tensor) -> (Tensor, f32) {
    // γ = (1/n) Σ |wᵢ|. f32 accumulation is fine for our toy sizes.
    let n = w.data.len() as f32;
    let abs_sum: f32 = w.data.iter().map(|x| x.abs()).sum();
    let gamma = abs_sum / n;

    // (γ + ε) so an all-zero W produces W_q = 0 (sane) instead of NaNs.
    let denom = gamma + TERNARY_EPS;

    // Round to nearest, clamp to {-1, 0, +1}. f32::round is half-away-from-zero,
    // symmetric across signs - doesn't bias the +1 vs -1 counts.
    let data = w
        .data
        .iter()
        .map(|x| (x / denom).round().clamp(-1.0, 1.0))
        .collect();

    (
        Tensor {
            data,
            shape: w.shape.clone(),
        },
        gamma,
    )
}

/// Absmax INT8 activation quantisation, per-token (BitNet b1.58 §2).
///
/// Per row of the 2D input:
///     α[i]     = max_j |x[i, j]|
///     x_q[i,j] = clamp(round(x[i, j] · 127 / α[i]), -128, +127)
///
/// Edge case: a row of all zeros has α = 0. Emit zeros directly to avoid NaN.
pub fn absmax_int8(x: &Tensor) -> (Tensor, Tensor) {
    assert_eq!(
        x.ndim(),
        2,
        "absmax_int8: x must be rank-2 [m, n], got rank {}",
        x.ndim()
    );
    let (m, n) = (x.shape[0], x.shape[1]);

    let mut x_q = vec![0.0_f32; m * n];
    let mut alpha = vec![0.0_f32; m];

    for i in 0..m {
        // Pass 1: row absmax.
        let mut a = 0.0_f32;
        for j in 0..n {
            let v = x.data[i * n + j].abs();
            if v > a {
                a = v;
            }
        }
        alpha[i] = a;
        if a == 0.0 {
            continue;
        }

        // Pass 2: scale, round, clamp into the [-128, 127] INT8 grid.
        let scale = 127.0 / a;
        for j in 0..n {
            x_q[i * n + j] = (x.data[i * n + j] * scale).round().clamp(-128.0, 127.0);
        }
    }

    (
        Tensor {
            data: x_q,
            shape: vec![m, n],
        },
        Tensor {
            data: alpha,
            shape: vec![m],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmean_ternary_clamps_and_zeroes_correctly() {
        let w = Tensor::from_vec(vec![2.0, -2.0, 0.0, 0.0], vec![1, 4]);
        let (w_q, gamma) = absmean_ternary(&w);
        assert!((gamma - 1.0).abs() < 1e-4, "γ ≈ 1.0, got {}", gamma);
        assert_eq!(w_q.data, vec![1.0, -1.0, 0.0, 0.0]);
        assert_eq!(w_q.shape, vec![1, 4]);
    }

    #[test]
    fn absmean_ternary_rounds_low_magnitude_to_zero() {
        let w = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]);
        let (w_q, gamma) = absmean_ternary(&w);
        assert!((gamma - 0.25).abs() < 1e-4, "γ ≈ 0.25, got {}", gamma);
        assert_eq!(w_q.data, vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn absmean_ternary_handles_all_zero_input() {
        let w = Tensor::zeros(vec![2, 3]);
        let (w_q, gamma) = absmean_ternary(&w);
        assert_eq!(gamma, 0.0);
        assert!(w_q.data.iter().all(|&v| v == 0.0));
        assert_eq!(w_q.shape, vec![2, 3]);
    }

    #[test]
    fn absmax_int8_per_row_scaling() {
        let x = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 0.1, -0.4, 0.2, 0.3], vec![2, 4]);
        let (x_q, alpha) = absmax_int8(&x);

        assert_eq!(alpha.shape, vec![2]);
        assert!((alpha.data[0] - 2.0).abs() < 1e-5);
        assert!((alpha.data[1] - 0.4).abs() < 1e-5);

        assert_eq!(x_q.shape, vec![2, 4]);
        assert_eq!(
            x_q.data,
            vec![64.0, 127.0, -64.0, 32.0, 32.0, -127.0, 64.0, 95.0]
        );
    }

    #[test]
    fn absmax_int8_handles_all_zero_row() {
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let (x_q, alpha) = absmax_int8(&x);

        assert_eq!(alpha.data[0], 0.0);
        assert_eq!(&x_q.data[0..3], &[0.0, 0.0, 0.0]);

        assert!((alpha.data[1] - 3.0).abs() < 1e-5);
        assert_eq!(&x_q.data[3..6], &[42.0, 85.0, 127.0]);
    }

    #[test]
    fn absmax_int8_clamps_at_extreme_grid_edges() {
        let x = Tensor::from_vec(vec![5.0, -5.0, 2.5], vec![1, 3]);
        let (x_q, alpha) = absmax_int8(&x);
        assert_eq!(alpha.data[0], 5.0);
        assert_eq!(x_q.data, vec![127.0, -127.0, 64.0]);
    }
}

//! Optimisers + learning-rate schedules + gradient clipping.
//!
//! Three pieces, all pure-Rust, no external deps:
//!   - `AdamW`: adaptive moment estimation with decoupled weight decay.
//!     The standard transformer optimiser since GPT-2.
//!   - `clip_grad_norm`: rescales gradients in-place so their global L2 norm
//!     does not exceed a given cap. Prevents exploding-gradient blow-ups.
//!   - `cosine_lr`: cosine schedule with linear warmup, the default LR
//!     trajectory for modern LLMs (smooth ramp up, smooth decay to a tiny
//!     floor, no spikes).
//!
//! All three operate on the `Model::for_each_param_with_grad` visitor or on
//! the leaves directly, so adding more optimisers later (Lion, Adafactor, etc.)
//! is a single new struct.

use crate::autograd::Var;
use crate::model::{Model, ModelLeaves};
use crate::tensor::Tensor;

// ---- AdamW ----

/// AdamW optimiser. State (`m`, `v`) is sized at construction time to match
/// the model's parameter list and stays constant across training steps.
///
/// Update per parameter `p` with gradient `g`:
/// ```text
///   m  = b1*m + (1 - b1)*g
///   v  = b2*v + (1 - b2)*g^2
///   m_hat  = m / (1 - b1^t)
///   v_hat  = v / (1 - b2^t)
///   p  = p - lr*(m_hat / (sqrt(v_hat) + eps) + wd*p)
/// ```
/// `wd * p` is the "decoupled" weight decay (added directly to the update,
/// not multiplied through the moments). LLaMA defaults: b1=0.9, b2=0.95, wd=0.1.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: u32,
    /// First-moment buffers, one tensor per model parameter (visitor order).
    m: Vec<Tensor>,
    /// Second-moment buffers, same length and shapes as `m`.
    v: Vec<Tensor>,
}

impl AdamW {
    /// Build an AdamW with state sized for `model`. Default hyperparameters
    /// match the LLaMA / BitNet pretraining recipe.
    pub fn new_for(model: &Model, lr: f32) -> Self {
        let shapes = model.param_shapes();
        let m = shapes.iter().map(|s| Tensor::zeros(s.clone())).collect();
        let v = shapes.iter().map(|s| Tensor::zeros(s.clone())).collect();
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            step_count: 0,
            m,
            v,
        }
    }

    /// Run one optimiser step. Mutates the model's master tensors in-place
    /// and updates the optimiser's running moments.
    pub fn step(&mut self, model: &mut Model, leaves: &ModelLeaves<'_>) {
        self.step_count += 1;
        let t = self.step_count as i32;
        let bc1 = 1.0 - self.beta1.powi(t);
        let bc2 = 1.0 - self.beta2.powi(t);

        // Local copies so the closure doesn't need to borrow self.
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        // Destructure to get split borrows for `m` and `v` separately.
        let m_slices = self.m.as_mut_slice();
        let v_slices = self.v.as_mut_slice();
        let mut idx = 0;

        model.for_each_param_with_grad(leaves, |p, g| {
            let m_i = &mut m_slices[idx];
            let v_i = &mut v_slices[idx];
            idx += 1;
            assert_eq!(p.data.len(), g.data.len());
            assert_eq!(p.data.len(), m_i.data.len());
            for j in 0..p.data.len() {
                let gj = g.data[j];
                m_i.data[j] = beta1 * m_i.data[j] + (1.0 - beta1) * gj;
                v_i.data[j] = beta2 * v_i.data[j] + (1.0 - beta2) * gj * gj;
                let m_hat = m_i.data[j] / bc1;
                let v_hat = v_i.data[j] / bc2;
                p.data[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * p.data[j]);
            }
        });
    }
}

// ---- gradient clipping ----

/// Compute the global L2 norm across every leaf gradient. If it exceeds
/// `max_norm`, scale every gradient cell by `max_norm / global_norm` so the
/// post-clip norm equals `max_norm` exactly. Returns the pre-clip norm so
/// callers can log it.
///
/// Operates on the leaf grad cells directly via `Tape::write_grad`. Must be
/// called AFTER `tape.backward(loss.id)` and BEFORE the optimiser step.
pub fn clip_grad_norm(leaves: &ModelLeaves<'_>, max_norm: f32) -> f32 {
    // Pass 1: compute global squared L2 norm.
    let mut sum_sq: f32 = 0.0;
    let mut accum = |g: &Tensor| {
        for &v in &g.data {
            sum_sq += v * v;
        }
    };
    accum(&leaves.token_embed.grad());
    accum(&leaves.pos_embed.grad());
    for lb in &leaves.blocks {
        for lh in &lb.heads {
            accum(&lh.w_q.grad());
            accum(&lh.w_k.grad());
            accum(&lh.w_v.grad());
            accum(&lh.w_o.grad());
        }
        accum(&lb.ffn_gate_w.grad());
        accum(&lb.ffn_up_w.grad());
        accum(&lb.ffn_down_w.grad());
    }
    accum(&leaves.lm_head.grad());

    let total_norm = sum_sq.sqrt();
    if total_norm <= max_norm || total_norm == 0.0 {
        return total_norm;
    }

    // Pass 2: rescale every leaf grad cell in place.
    let scale = max_norm / total_norm;
    rescale_leaf_grad(leaves.token_embed, scale);
    rescale_leaf_grad(leaves.pos_embed, scale);
    for lb in &leaves.blocks {
        for lh in &lb.heads {
            rescale_leaf_grad(lh.w_q, scale);
            rescale_leaf_grad(lh.w_k, scale);
            rescale_leaf_grad(lh.w_v, scale);
            rescale_leaf_grad(lh.w_o, scale);
        }
        rescale_leaf_grad(lb.ffn_gate_w, scale);
        rescale_leaf_grad(lb.ffn_up_w, scale);
        rescale_leaf_grad(lb.ffn_down_w, scale);
    }
    rescale_leaf_grad(leaves.lm_head, scale);

    total_norm
}

/// Rescale a leaf's gradient in place by `scale`. Uses `Tape::write_grad` so
/// `optim` doesn't need direct access to the inner `RefCell`.
fn rescale_leaf_grad(var: Var<'_>, scale: f32) {
    let mut g = var.grad();
    for v in &mut g.data {
        *v *= scale;
    }
    var.tape.write_grad(var.id, g);
}

// ---- cosine LR schedule with warmup ----

/// Cosine learning-rate schedule with linear warmup.
///
/// - Steps `0..warmup`: linear ramp from 0 to `peak`.
/// - Steps `warmup..total`: cosine decay from `peak` to `floor`.
/// - Steps `>= total`: clamp at `floor`.
pub fn cosine_lr(step: usize, warmup: usize, total: usize, peak: f32, floor: f32) -> f32 {
    if step < warmup {
        return peak * (step as f32) / (warmup.max(1) as f32);
    }
    if step >= total {
        return floor;
    }
    let progress = (step - warmup) as f32 / ((total - warmup).max(1) as f32);
    let cos_term = (1.0 + (std::f32::consts::PI * progress).cos()) * 0.5;
    floor + (peak - floor) * cos_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tape;
    use crate::model::ModelConfig;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 8,
            hidden_dim: 4,
            n_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            max_seq_len: 4,
            n_blocks: 2,
        }
    }

    #[test]
    fn adamw_state_matches_param_shapes() {
        let cfg = tiny_config();
        let model = Model::new(&cfg, 0);
        let opt = AdamW::new_for(&model, 1e-3);
        let shapes = model.param_shapes();
        assert_eq!(opt.m.len(), shapes.len());
        assert_eq!(opt.v.len(), shapes.len());
        for (i, s) in shapes.iter().enumerate() {
            assert_eq!(opt.m[i].shape, *s);
            assert_eq!(opt.v[i].shape, *s);
        }
    }

    fn loss_on(model: &Model, ids: &[usize], targets: &[usize]) -> f32 {
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);
        let logits = model.forward(&leaves, ids);
        logits.cross_entropy(targets).value().data[0]
    }

    #[test]
    fn adamw_step_decreases_loss_on_a_fixed_window() {
        // Small model, fixed window, tame LR + small weight decay so the test
        // is robust against STE-quantisation noise (which is real and noisy at
        // this scale, see model.rs `training_steps_reduce_loss_...`).
        let cfg = tiny_config();
        let mut model = Model::new(&cfg, 7);
        let mut opt = AdamW::new_for(&model, 1e-2);
        opt.weight_decay = 0.01;

        let ids = vec![1, 2, 3, 4];
        let targets = vec![2, 3, 4, 5];

        let l0 = loss_on(&model, &ids, &targets);

        let mut min_seen = l0;
        for _ in 0..150 {
            let tape = Tape::new();
            let leaves = model.register_leaves(&tape);
            let logits = model.forward(&leaves, &ids);
            let loss = logits.cross_entropy(&targets);
            let lv = loss.value().data[0];
            if lv < min_seen {
                min_seen = lv;
            }
            tape.backward(loss.id);
            opt.step(&mut model, &leaves);
        }

        // STE training is noisy step-to-step; assert "best loss seen during
        // training" rather than "final loss." Same property the M9 test asserts.
        assert!(
            min_seen < l0 * 0.7,
            "AdamW failed to reduce loss enough: {} -> min seen {}",
            l0,
            min_seen
        );
    }

    #[test]
    fn clip_grad_norm_caps_at_max_when_above_threshold() {
        let cfg = tiny_config();
        let model = Model::new(&cfg, 0);
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);

        let logits = model.forward(&leaves, &[1, 2, 3, 4]);
        let loss = logits.cross_entropy(&[2, 3, 4, 5]);
        tape.backward(loss.id);

        let pre = clip_grad_norm(&leaves, 0.1);
        assert!(pre > 0.0);

        let mut post_sq = 0.0_f32;
        model.for_each_grad(&leaves, |g| {
            for &v in &g.data {
                post_sq += v * v;
            }
        });
        let post = post_sq.sqrt();
        assert!(
            (post - 0.1).abs() < 1e-3,
            "post-clip norm {} not close to 0.1 (pre = {})",
            post,
            pre
        );
    }

    #[test]
    fn clip_grad_norm_noop_when_already_under_cap() {
        let cfg = tiny_config();
        let model = Model::new(&cfg, 0);
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);

        let logits = model.forward(&leaves, &[1, 2, 3, 4]);
        let loss = logits.cross_entropy(&[2, 3, 4, 5]);
        tape.backward(loss.id);

        let mut pre_sq = 0.0_f32;
        model.for_each_grad(&leaves, |g| {
            for &v in &g.data {
                pre_sq += v * v;
            }
        });
        let pre = pre_sq.sqrt();

        let big = pre * 10.0;
        let reported = clip_grad_norm(&leaves, big);
        assert!((reported - pre).abs() < 1e-5);

        let mut post_sq = 0.0_f32;
        model.for_each_grad(&leaves, |g| {
            for &v in &g.data {
                post_sq += v * v;
            }
        });
        let post = post_sq.sqrt();
        assert!((post - pre).abs() < 1e-4);
    }

    #[test]
    fn cosine_lr_schedule_shape() {
        assert!((cosine_lr(0, 100, 1000, 1.0, 0.1) - 0.0).abs() < 1e-6);
        assert!((cosine_lr(50, 100, 1000, 1.0, 0.1) - 0.5).abs() < 1e-6);
        assert!((cosine_lr(100, 100, 1000, 1.0, 0.1) - 1.0).abs() < 1e-6);
        let mid = cosine_lr(550, 100, 1000, 1.0, 0.1);
        assert!((mid - 0.55).abs() < 1e-2);
        assert!((cosine_lr(2000, 100, 1000, 1.0, 0.1) - 0.1).abs() < 1e-6);
    }
}

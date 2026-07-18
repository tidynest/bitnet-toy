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

use crate::model::Model;
use crate::tensor::Tensor;

// ---- AdamW state for resume ----

/// Persistent snapshot of an `AdamW` optimiser's mutable state. Carried
/// through the on-disk checkpoint so resumes don't lose AdamW's momentum
/// estimates - the first ~30 steps after a snapshot-less resume were wobbly
/// while `m` / `v` re-established from zeros.
///
/// Lives outside `AdamW` itself so the export/import code can construct it
/// from on-disk bytes without needing the live optimiser.
#[derive(Debug, Clone)]
pub struct OptimState {
    pub step_count: u32,
    pub m: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

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

    /// Snapshot the optimiser's mutable state for export. Cloning the moment
    /// buffers is cheap relative to one training step; if it ever shows up in
    /// profiling, change the export path to take `&self` and stream tensors.
    pub fn snapshot(&self) -> OptimState {
        OptimState {
            step_count: self.step_count,
            m: self.m.clone(),
            v: self.v.clone(),
        }
    }

    /// Replace the optimiser's mutable state with a previously-snapshotted
    /// `OptimState`. Asserts that every shape matches the optimiser's existing
    /// buffers - a mismatch means the state was saved against a different
    /// model architecture and the per-tensor AdamW math would silently corrupt.
    pub fn restore(&mut self, state: OptimState) {
        assert_eq!(
            state.m.len(),
            self.m.len(),
            "OptimState m length {} doesn't match optimiser {}",
            state.m.len(),
            self.m.len()
        );
        for (i, (existing, loaded)) in self.m.iter().zip(&state.m).enumerate() {
            assert_eq!(
                existing.shape, loaded.shape,
                "OptimState m[{}] shape mismatch",
                i
            );
        }
        for (i, (existing, loaded)) in self.v.iter().zip(&state.v).enumerate() {
            assert_eq!(
                existing.shape, loaded.shape,
                "OptimState v[{}] shape mismatch",
                i
            );
        }
        self.step_count = state.step_count;
        self.m = state.m;
        self.v = state.v;
    }

    /// Run one optimiser step from pre-collected gradient tensors.
    /// `grads` must be in the same canonical visitor order as
    /// `Model::for_each_param_mut`. The training loop calls this once per
    /// step with averaged gradients (batched path) or with single-window
    /// gradients (when `batch_size == 1`); callers without a tape pull
    /// gradients themselves via `Model::for_each_grad`.
    pub fn step_with_grads(&mut self, model: &mut Model, grads: &[Tensor]) {
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

        let m_slices = self.m.as_mut_slice();
        let v_slices = self.v.as_mut_slice();
        let bf16 = crate::tensor::bf16_masters_enabled();
        let mut idx = 0;

        model.for_each_param_mut(|p| {
            let g = &grads[idx];
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
                // Issue #23: masters are STORED at bf16 precision - the
                // update computes in f32, the store narrows (RNE). The
                // moments above stay full f32.
                if bf16 {
                    p.data[j] = crate::tensor::narrow_to_bf16(p.data[j]);
                }
            }
        });
    }
}

// ---- gradient clipping ----

/// Compute the global L2 norm across a collected gradient slice. If it
/// exceeds `max_norm`, rescale every cell in place so the post-clip norm
/// equals `max_norm` exactly. Returns the pre-clip norm.
///
/// Used by the batched training path: after gradients are averaged across
/// the batch, this clips them once before the optimiser step. The shape of
/// the slice doesn't matter to this function - it walks the flat data.
pub fn clip_grad_norm_tensors(grads: &mut [Tensor], max_norm: f32) -> f32 {
    let mut sum_sq: f32 = 0.0;
    for g in grads.iter() {
        for &v in &g.data {
            sum_sq += v * v;
        }
    }
    let total_norm = sum_sq.sqrt();
    if total_norm <= max_norm || total_norm == 0.0 {
        return total_norm;
    }
    let scale = max_norm / total_norm;
    for g in grads.iter_mut() {
        for v in g.data.iter_mut() {
            *v *= scale;
        }
    }
    total_norm
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

    /// Issue #23: after any optimiser step every master sits exactly on
    /// the bf16 grid (low 16 bits zero) while the moments stay full f32.
    #[test]
    fn adamw_step_leaves_masters_on_bf16_grid() {
        if !crate::tensor::bf16_masters_enabled() {
            eprintln!("skipping: BITNET_BF16_MASTERS=0");
            return;
        }
        let cfg = tiny_config();
        let mut model = Model::new(&cfg, 3);
        let mut opt = AdamW::new_for(&model, 1e-2);
        let grads: Vec<Tensor> = model
            .param_shapes()
            .into_iter()
            .map(|shape| {
                let len: usize = shape.iter().product();
                Tensor {
                    data: (0..len).map(|i| (i as f32 * 0.37).sin()).collect(),
                    shape,
                }
            })
            .collect();
        opt.step_with_grads(&mut model, &grads);
        model.for_each_param_mut(|t| {
            for (j, v) in t.data.iter().enumerate() {
                assert_eq!(
                    v.to_bits() & 0xFFFF,
                    0,
                    "master cell {j} off the bf16 grid: {v}"
                );
            }
        });
        // Moments untouched by narrowing: at least one has live low bits.
        let snap = opt.snapshot();
        let any_full_precision = snap
            .m
            .iter()
            .chain(&snap.v)
            .flat_map(|t| &t.data)
            .any(|v| v.to_bits() & 0xFFFF != 0);
        assert!(
            any_full_precision,
            "moments appear narrowed - they must stay f32"
        );
    }

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

    /// Run forward + backward once and collect every leaf gradient into a
    /// `Vec<Tensor>` in canonical visitor order. Used by the optim tests
    /// after the leaf-driven helpers were retired in favour of the
    /// tensor-driven path.
    fn collect_grads(model: &Model, ids: &[usize], targets: &[usize]) -> Vec<Tensor> {
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);
        let logits = model.forward(&leaves, ids);
        let loss = logits.cross_entropy(targets);
        tape.backward(loss.id);
        let mut grads = Vec::new();
        model.for_each_grad(&leaves, |g| grads.push(g.clone()));
        grads
    }

    fn norm_of(grads: &[Tensor]) -> f32 {
        let mut sum_sq = 0.0_f32;
        for g in grads {
            for &v in &g.data {
                sum_sq += v * v;
            }
        }
        sum_sq.sqrt()
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
            let lv = loss_on(&model, &ids, &targets);
            if lv < min_seen {
                min_seen = lv;
            }
            let grads = collect_grads(&model, &ids, &targets);
            opt.step_with_grads(&mut model, &grads);
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
        let mut grads = collect_grads(&model, &[1, 2, 3, 4], &[2, 3, 4, 5]);

        let pre = clip_grad_norm_tensors(&mut grads, 0.1);
        assert!(pre > 0.0);

        let post = norm_of(&grads);
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
        let mut grads = collect_grads(&model, &[1, 2, 3, 4], &[2, 3, 4, 5]);

        let pre = norm_of(&grads);
        let big = pre * 10.0;
        let reported = clip_grad_norm_tensors(&mut grads, big);
        assert!((reported - pre).abs() < 1e-5);

        let post = norm_of(&grads);
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

    #[test]
    fn cosine_lr_resume_continuation_skips_warmup_and_decays_smoothly() {
        // The v0.14 resume pattern: pass `step + offset` and a stretched
        // `total = offset + n_steps`. The schedule should deliver a
        // continuation LR somewhere between peak and floor at the start of
        // the new run (post-warmup territory) and reach floor at the end.
        // Concretely: a 10k+10k resume should land at the cosine half-way
        // point at the resume's step 0.
        let warmup = 200;
        let peak = 3e-3;
        let floor = 3e-4;
        let offset = 10_000;
        let n_steps = 10_000;
        let total = offset + n_steps; // 20_000

        // Resume's step 0 (effective 10_000): post-warmup, halfway through
        // a 19_800-step cosine band. progress ~ (10_000 - 200) / 19_800 ~ 0.495.
        // LR ~ floor + 0.5 * (peak - floor) * (1 + cos(pi * 0.495))
        //    ~ floor + 0.5 * peak * 0.0157  -> just above midpoint.
        let lr_start = cosine_lr(offset, warmup, total, peak, floor);
        assert!(
            lr_start > floor && lr_start < peak,
            "resume step 0 LR {} outside (floor {}, peak {})",
            lr_start,
            floor,
            peak
        );
        // Resume's mid (effective 15_000): roughly 3/4 down the cosine.
        let lr_mid = cosine_lr(5000 + offset, warmup, total, peak, floor);
        assert!(lr_mid < lr_start, "LR should still be decaying mid-resume");
        // Resume's last step: at the cosine's tail, very close to floor.
        let lr_end = cosine_lr(n_steps - 1 + offset, warmup, total, peak, floor);
        assert!(
            (lr_end - floor).abs() / floor < 0.02,
            "resume step {} LR {} should be ~floor {} (relative err > 2%)",
            n_steps - 1,
            lr_end,
            floor
        );
        // Sanity: with offset = 0 (fresh run) the schedule reduces to the
        // pre-v0.14 behaviour exactly.
        let lr_fresh_step_0 = cosine_lr(0, warmup, n_steps, peak, floor);
        assert!((lr_fresh_step_0 - 0.0).abs() < 1e-6);
    }
}

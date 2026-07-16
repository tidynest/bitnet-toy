//! M4 entry point: 1D linear regression by SGD, demonstrating that the
//! hand-rolled autograd in src/autograd.rs actually drives loss downward.
//!
//! Task:   fit  y = 2x  to 8 hand-picked points.
//! Model:  pred = x · w     (single scalar weight, no bias - bias broadcast
//!                            isn't worth the complexity until something else
//!                            forces it; M7 attention will).
//! Loss:    mean((pred − y)²)
//! Update:  w <- w - lr · dL/dw   (vanilla SGD)

// Crate-level clippy allows. Pedantic surfaces a lot of stylistic noise that
// isn't worth fixing in a research / learning codebase; the items left to fail
// are the ones that catch actual bugs.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::float_cmp)]
// Default-group lints kept off for the same reason: in tensor / autograd math,
// index-parallel loops and wide operand lists read clearer than the iterator
// rewrites clippy suggests, and the tape/closure types are inherent to the design.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod attention;
mod autograd;
mod bitlinear;
mod block;
#[cfg(feature = "cuda")]
mod cuda;
mod data;
mod device;
mod export;
mod ffn;
mod inference;
mod inference_kv;
mod model;
mod optim;
mod tensor;

use autograd::{Tape, Var};
use tensor::Tensor;

/// One full training run. Returns `(initial_loss, final_loss, final_w)` so the
/// integration test below can sanity-check convergence without parsing stdout.
fn train_linear_regression(n_steps: usize, lr: f32) -> (f32, f32, f32) {
    // ── Synthetic data: 8 points exactly on  y = 2x  (no noise, no offset). ──
    let x_data = Tensor::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], vec![8, 1]);
    let y_data = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![8, 1]);

    // Master weight: lives OUTSIDE any tape. Each step we will clone its current
    // value into a fresh tape as a leaf, then multiply this master after backward.
    // Started at zero - for 1D regression that's fine because the gradient at
    // zero points unambiguously toward the global minimum.
    let mut w = Tensor::from_vec(vec![0.0], vec![1, 1]);

    let mut initial_loss: Option<f32> = None;
    let mut final_loss = 0.0_f32;

    for step in 0..n_steps {
        // Fresh tape every iteration. The old tape - including all saved input
        // tensors captured by backward closures - is released when this scope
        // ends, so memory is bounded by ONE step's worth of ops, not the full run.
        let tape = Tape::new();

        // Register inputs, target, and the current weight as leaves of this tape.
        // The data is cloned IN; the master `w` continues to live outside.
        let x_var = Var::leaf(&tape, x_data.clone());
        let y_var = Var::leaf(&tape, y_data.clone());
        let w_var = Var::leaf(&tape, w.clone());

        // Forward:
        //   pred = x · w           shape [8, 1]
        //   diff = pred − y        shape [8, 1]
        //   sq   = diff ⊙ diff     shape [8, 1]   (square via reused leaf;
        //                                          accumulation gives ∂sq/∂diff = 2·diff)
        //   loss = mean(sq)        shape [1]      (scalar - clean seed for backward)
        let pred = x_var.matmul(w_var);
        let diff = pred.sub(y_var);
        let sq = diff.mul(diff);
        let loss = sq.mean();

        let loss_value = loss.value().data[0];
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = loss_value;

        // Backward - fills `w_var.grad()` with ∂loss/∂w.
        tape.backward(loss.id);

        // SGD update on the MASTER (not on w_var, which is about to die with the tape).
        //   w_master ← w_master − lr · ∂L/∂w
        let g = w_var.grad();
        for i in 0..w.data.len() {
            w.data[i] -= lr * g.data[i];
        }

        // Periodic log so convergence is visible to the eye.
        if step % 20 == 0 || step == n_steps - 1 {
            println!(
                "step {:>4}   loss = {:.6e}   w = {:.6}",
                step, loss_value, w.data[0]
            );
        }
        // `tape` is dropped here - entire computational graph released.
    }

    (initial_loss.unwrap(), final_loss, w.data[0])
}

/// M6 gate: BitLinear-shaped training on `y = 2x₁ − 2x₂`.
///
/// Same training-loop structure as `train_linear_regression`, but the forward
/// graph runs through STE-quantised weights and per-row STE-quantised
/// activations. If the gradient successfully threads back through the quantisers
/// (i.e. STE works), `master_w` converges to ≈ (2, −2). If STE is broken, every
/// gradient through the quantiser is zero and `master_w` stays frozen at zeros.
///
/// Returns `(initial_loss, final_loss, [w0, w1])`.
fn train_bitlinear_regression(n_steps: usize, lr: f32) -> (f32, f32, Vec<f32>) {
    // ── Synthetic data ──
    // 4 rows, 2 features, single output. Rows include both axis-aligned cases
    // (1,0) and (0,1) and "mixed" cases (2,1), (1,2) so the per-row activation
    // quantisation gets exercised at different α magnitudes.
    let x_data = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0], vec![4, 2]);
    // y = 2 · x1  −  2 · x2  (exactly hits the ternary grid at γ=2, W_q=[+1,−1]).
    let y_data = Tensor::from_vec(vec![2.0, -2.0, 2.0, -2.0], vec![4, 1]);

    // Master weight: shape [in, out] = [2, 1].
    // Zeros init is fine - STE pushes off zero on the first step (W_eff=0 still
    // produces a non-zero gradient via the matmul-of-x part of the chain).'
    let mut master_w = Tensor::from_vec(vec![0.0, 0.0], vec![2, 1]);

    let mut initial_loss: Option<f32> = None;
    let mut final_loss = 0.0_f32;

    for step in 0..n_steps {
        let tape = Tape::new();
        let x_var = Var::leaf(&tape, x_data.clone());
        let y_var = Var::leaf(&tape, y_data.clone());
        let w_var = Var::leaf(&tape, master_w.clone());

        // BitLinear-shaped forward graph:
        //   x_eff = quantise_acts_ste(x)        per-row INT8 dequant, identity backward
        //   w_eff = quantise_weights_ste(w)     ternary dequant,      identity backward
        //   pred  = x_eff · w_eff               plain matmul on dequantised values
        //   loss  = mean((pred − y)²)
        //
        // Note: w_var is stored as [in, out] = [2, 1], matching matmul's expectations
        // directly - no transpose needed here. (Var::transpose_2d exists from M6 portion 1
        // for cases where you start from a [out, in] layout, e.g. if/when we wrap
        // the BitLinear *struct* itself onto the tape.)
        let x_eff = x_var.quantise_acts_ste();
        let w_eff = w_var.quantise_weights_ste();
        let pred = x_eff.matmul(w_eff);
        let diff = pred.sub(y_var);
        let sq = diff.mul(diff);
        let loss = sq.mean();

        let loss_value = loss.value().data[0];
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = loss_value;

        tape.backward(loss.id);

        // SGD on master_w. Gradient comes through STE - the quantiser's
        // identity-backward is what makes this nonzero.
        let g = w_var.grad();
        for i in 0..master_w.data.len() {
            master_w.data[i] -= lr * g.data[i];
        }

        if step % 50 == 0 || step == n_steps - 1 {
            println!(
                "step {:>4}   loss = {:.6e}   w = [{:>+8.4}, {:>+8.4}]",
                step, loss_value, master_w.data[0], master_w.data[1]
            );
        }
    }

    (initial_loss.unwrap(), final_loss, master_w.data.clone())
}

/// Hyperparameters for one full BitNet LM training run. Sensible defaults
/// for the embedded TINY_CORPUS demo; override fields for real corpora.
pub struct TrainConfig {
    pub n_steps: usize,
    pub peak_lr: f32,
    pub floor_lr: f32,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub weight_decay: f32,
    pub adamw_beta1: f32,
    pub adamw_beta2: f32,
    pub seed: u64,
    pub model: crate::model::ModelConfig,
    /// If `Some(path)`, train on the file's contents instead of `TINY_CORPUS`.
    pub corpus_path: Option<std::path::PathBuf>,
    /// Print a status line every `log_every` steps (and on the final step).
    pub log_every: usize,
    /// Fraction of the corpus held out for validation. The split is taken
    /// from the END of the encoded id stream so all training windows precede
    /// all validation windows; no leakage. Set to 0.0 to disable validation
    /// entirely (training metrics only - useful for the embedded TINY_CORPUS
    /// demo where the corpus is too short to spare any tokens).
    pub val_split: f32,
    /// Run a held-out validation pass every `eval_every` steps. The pass
    /// samples a deterministic, evenly-spaced subset of validation windows
    /// (size `val_eval_samples`), so the reported val_ppl is comparable
    /// across eval points. Set to 0 to disable periodic validation.
    pub eval_every: usize,
    /// Number of validation windows used per evaluation pass. Held constant
    /// across eval points so val_ppl trends are meaningful. Lower = faster
    /// eval, noisier val_ppl. 100-200 is a good range for most corpora.
    pub val_eval_samples: usize,
    /// Number of training windows processed per optimiser step. With
    /// `batch_size = 1` (default), training matches the older single-window
    /// path exactly. With `batch_size > 1`, every step samples `batch_size`
    /// independent windows, computes per-window gradients, averages them,
    /// applies one optimiser update. Larger batches give smoother gradient
    /// estimates and need fewer steps to reach the same val_ppl, but each
    /// step costs `batch_size`x more compute (parallelisable - see `n_workers`).
    pub batch_size: usize,
    /// Maximum threads used by the batched step. With `n_workers >= batch_size`
    /// every window in the batch processes in parallel. Higher = more wall-clock
    /// speedup but heavier sustained CPU load (matters for thermals on laptops).
    /// Set to 1 to force serial-batched mode for deterministic comparison.
    /// Set to 0 to disable parallelism even when `batch_size > 1` (same as 1
    /// in practice). On the 7940HS, `n_workers = 4` keeps the chip below
    /// throttling territory; bump to 8 if your fans cope.
    pub n_workers: usize,
    /// If `Some(model)`, start training from these weights instead of fresh
    /// random init. The model's `ModelConfig` overrides `self.model` so the
    /// optimiser sizes and forward shapes match the loaded weights exactly.
    /// LR schedule warmup restarts fresh; resumes get clean momentum if
    /// `start_from_optim` is also set.
    pub start_from: Option<crate::model::Model>,
    /// If `Some(state)` (and `start_from` is also `Some`), restore the
    /// AdamW optimiser's momentum / variance buffers from this snapshot.
    /// Pre-v0.7 checkpoints don't carry optim state; resuming from one of
    /// those leaves this `None` and the optimiser starts from zeros, costing
    /// the usual ~30 wobbly steps after resume.
    pub start_from_optim: Option<crate::optim::OptimState>,
    /// Starting offset into the cosine LR schedule. The training loop uses
    /// `cosine_lr(step + start_step_offset, warmup_steps, cosine_total_steps,
    /// peak_lr, floor_lr)`. Default 0 (fresh start).
    ///
    /// When resuming with `start_from_optim = Some(state)`, set this to
    /// `state.step_count` so the LR schedule picks up where it left off
    /// instead of restarting at the warmup floor of zero. Without this
    /// offset (the v0.7-v0.13 default), every `cargo run` re-walks the
    /// 200-step warmup ramp at peak LR, which perturbs an already-converged
    /// model and was diagnosed as the v0.13 cumulative-30k plateau cause.
    pub start_step_offset: usize,
    /// Override for the cosine schedule's `total` argument. `None` means
    /// "use `n_steps`" (the v0.7-v0.13 behaviour). When resuming, set to
    /// `start_step_offset + n_steps` so the cosine decay denominator
    /// stretches across both the prior training and the new continuation,
    /// giving a smooth tail from previous-run-LR down to floor at the end
    /// of the new run.
    pub cosine_total_steps: Option<usize>,
    /// Phase 5.a: route gradient computation through the GPU
    /// `CudaModel::compute_grads_for_window_bitnet` instead of the CPU
    /// autograd `compute_grads_for_window`. Both paths produce
    /// gradients in the same canonical visitor order; the GPU path
    /// applies BitNet ternary STE quantisation matching the autograd
    /// CPU path's math (`Var::quantise_acts_ste` +
    /// `Var::quantise_weights_ste`). Requires the binary to be built
    /// with `--features cuda`; if the flag is set without the feature
    /// the training loop exits with a clear error.
    ///
    /// Default: `false` (CPU autograd path, the v0.1-v0.18 behaviour).
    /// The device-resident model is built once at step 0 and then
    /// refreshed in place per step via `CudaModel::sync_from_cpu`
    /// (issue #1) - pure H->D copies, no per-step device allocations.
    pub use_cuda_backward: bool,
}

impl TrainConfig {
    /// Defaults targeting the embedded TINY_CORPUS demo: small model, 300
    /// steps, AdamW with cosine LR + grad clipping.
    pub fn tiny_demo() -> Self {
        Self {
            n_steps: 300,
            peak_lr: 5e-3,
            floor_lr: 5e-4,
            warmup_steps: 30,
            grad_clip: 1.0,
            weight_decay: 0.05,
            adamw_beta1: 0.9,
            adamw_beta2: 0.95,
            seed: 1337,
            model: crate::model::ModelConfig {
                vocab_size: 0, // overridden after vocab is built
                hidden_dim: 16,
                // 2 heads * 8 head_dim == 16 = hidden_dim. Keeps total
                // attention parameter count identical to the old single-head
                // (head_dim 16) model while exposing the head-loop summation.
                n_heads: 2,
                head_dim: 8,
                ffn_dim: 32,
                max_seq_len: 16,
                n_blocks: 2,
            },
            corpus_path: None,
            log_every: 50,
            // TINY_CORPUS is too short to spare any tokens for held-out
            // validation; running on it would leave only a handful of
            // windows on each side and the val signal would be useless.
            val_split: 0.0,
            eval_every: 0,
            val_eval_samples: 0,
            // Tiny demo runs serial single-window so timings stay tight and
            // tests remain deterministic.
            batch_size: 1,
            n_workers: 1,
            start_from: None,
            start_from_optim: None,
            start_step_offset: 0,
            cosine_total_steps: None,
            use_cuda_backward: false,
        }
    }

    /// Defaults targeting the full TinyShakespeare corpus at `data/tinyshakespeare.txt`.
    /// Bigger model, longer sequences, more steps. Expect ~25-35 minutes on CPU.
    ///
    /// Sizing rationale (v0.13 bump from the v0.9-v0.12 ~2M-param config):
    ///   - v0.12 hit val_ppl 5.235 in 10k steps and 5.034 cumulative across
    ///     20k steps. Trajectory still descending - capacity, not training,
    ///     is the bottleneck.
    ///   - 1.5x the hidden_dim and ffn_dim, 1.5x the n_heads (head_dim still 16,
    ///     the BitNet b1.58 per-head dimensionality), n_blocks unchanged at 6.
    ///     Total parameter count lands around 5M (~2.5x v0.12).
    ///   - n_steps stays at 10_000. The bigger model converges faster per
    ///     parameter at this scale; resume to 20k or 30k if the curve still
    ///     looks promising at the end.
    ///   - warmup_steps stays at 200 (a fixed-budget LR ramp).
    pub fn shakespeare() -> Self {
        Self {
            n_steps: 10_000,
            peak_lr: 3e-3,
            floor_lr: 3e-4,
            warmup_steps: 200,
            grad_clip: 1.0,
            weight_decay: 0.1,
            adamw_beta1: 0.9,
            adamw_beta2: 0.95,
            seed: 1337,
            model: crate::model::ModelConfig {
                vocab_size: 0, // filled after vocab is built
                hidden_dim: 192,
                // 12 heads * 16 head_dim == 192 = hidden_dim. Keeping head_dim
                // at 16 (the BitNet b1.58 per-head dimensionality); the extra
                // hidden capacity is spent on more orthogonal attention
                // subspaces, not on bigger per-head dims. RoPE rotates pairs
                // within head_dim, so head_dim must stay even - 16 is fine.
                n_heads: 12,
                head_dim: 16,
                ffn_dim: 384,
                max_seq_len: 64,
                n_blocks: 6,
            },
            corpus_path: Some(std::path::PathBuf::from("data/tinyshakespeare.txt")),
            log_every: 100,
            // 10 percent val split: with TinyShakespeare's ~1.1M chars,
            // that leaves ~111K val chars, ~1.7K val windows at seq_len 64.
            val_split: 0.10,
            // Eval every 500 steps gives 10 measurements over a 5000-step
            // run, which is plenty to see the val_ppl trajectory without
            // adding meaningful overhead.
            eval_every: 500,
            // 100 windows per eval is a fixed-budget tradeoff: ~1 second
            // wall-clock per eval, val_ppl noise floor around 1-2%.
            val_eval_samples: 100,
            // Batched training: 4 windows per step, parallelised across
            // `n_workers` threads via `std::thread::scope`. Each worker runs
            // one full window's forward + backward; matmul stays serial
            // (with AVX2 SIMD) inside the worker because at this model scale
            // every individual matmul (1k-16k output elements) is too small
            // for the per-call thread-spawn cost to amortise. Outer-level
            // threading is the right granularity. Bump `BITNET_MATMUL_THREADS`
            // to N>1 if you scale the model up enough that a single matmul
            // becomes the dominant per-step work; do not stack both levels.
            batch_size: 4,
            n_workers: 4,
            start_from: None,
            start_from_optim: None,
            start_step_offset: 0,
            cosine_total_steps: None,
            use_cuda_backward: false,
        }
    }

    /// Bigger sibling of `shakespeare()`: ~8.5M parameters, sequence length
    /// doubled to 128. Aimed at squeezing the most out of the RoPE / SwiGLU
    /// architecture before the next training tier (multi-block + multi-head
    /// scaling beyond what fits comfortably on one CPU).
    ///
    /// Sizing (vs v0.13):
    ///   - `hidden_dim` 192 -> 256
    ///   - `ffn_dim` 384 -> 1024 (4x hidden, the standard transformer
    ///     expansion factor; v0.13 was at 2x because we were keeping the
    ///     param-count budget tight)
    ///   - `n_heads` 12 -> 16, `head_dim` still 16 (so 16 * 16 == 256)
    ///   - `n_blocks` 6 -> 8
    ///   - `max_seq_len` 64 -> 128 (RoPE handles arbitrary seq_len natively;
    ///     longer context is where rotary positional info pays the most)
    ///
    /// Wall-clock cost on a 7940HS with the v0.11 SIMD kernel: roughly 4-5x
    /// per step over v0.13, so a 10k-step run is ~2-2.5 hours. Plan for an
    /// overnight or multi-resume training. The v0.14 LR fix means resumes
    /// continue the cosine schedule cleanly, no warmup-restart penalty.
    pub fn shakespeare_large() -> Self {
        let mut cfg = Self::shakespeare();
        cfg.model = crate::model::ModelConfig {
            vocab_size: 0, // filled after vocab is built
            hidden_dim: 256,
            n_heads: 16,
            head_dim: 16, // 16 * 16 == 256
            ffn_dim: 1024,
            max_seq_len: 128,
            n_blocks: 8,
        };
        // Tuning carried forward from v0.16.x lessons:
        //   - At seq_len 128 individual per-window gradients carry 2x the
        //     attention noise of seq_len 64 (more positions, more entropy in
        //     the softmax during early training). Bumping `batch_size` from
        //     4 to 16 averages that out at ~4x compute per step but reduces
        //     gradient variance enough that the same n_steps reaches a
        //     better val_ppl. n_workers stays at 4 - the GPU path serialises
        //     workers anyway, and the CPU path's matmul threading is the
        //     productive parallelism layer at this scale.
        //   - Warmup 200 -> 500. The bigger model has more parameters whose
        //     adamw moments need to settle; a longer warmup ramp avoids
        //     early-step LR overshoot that the v0.13 config got away with at
        //     ~5M params but starts to bite at ~8.5M.
        cfg.batch_size = 16;
        cfg.warmup_steps = 500;
        cfg
    }
}

/// Run forward + backward on a single training window and return the gradient
/// set (in canonical visitor order) plus the scalar loss. The tape is built
/// and dropped inside this call, so memory stays bounded by one window's worth
/// of recorded ops.
///
/// This is the per-window worker used by the batched training path - both the
/// serial single-thread case and the parallel `std::thread::scope` case call
/// it. Keeping all the autograd graph construction in one tight function
/// makes the parallel-vs-serial split a pure question of how the windows are
/// dispatched, not how each one is computed.
fn compute_grads_for_window(
    model: &crate::model::Model,
    input: &[usize],
    target: &[usize],
) -> (Vec<crate::tensor::Tensor>, f32) {
    let tape = Tape::new();
    let leaves = model.register_leaves(&tape);
    let logits = model.forward(&leaves, input);
    let loss = logits.cross_entropy(target);
    let loss_val = loss.value().data[0];
    tape.backward(loss.id);

    let mut grads = Vec::with_capacity(model.param_shapes().len());
    model.for_each_grad(&leaves, |g| grads.push(g.clone()));
    (grads, loss_val)
}

/// Compute averaged gradients across `batch_size` randomly-sampled training
/// windows. Returns the averaged gradient set and the mean loss. Up to
/// `n_workers` threads run forward+backward in parallel via
/// `std::thread::scope`; setting `n_workers` to 1 falls back to a serial loop
/// that is byte-for-byte deterministic given the same RNG state.
///
/// The window indices are sampled up front from a single RNG before any
/// threads spawn, so changing `n_workers` does not change which windows the
/// batch sees - only the wall-clock cost of processing them.
fn compute_batched_grads(
    model: &crate::model::Model,
    windows: &[(Vec<usize>, Vec<usize>)],
    rng: &mut crate::data::Lcg,
    batch_size: usize,
    n_workers: usize,
) -> (Vec<crate::tensor::Tensor>, f32) {
    assert!(batch_size >= 1, "batch_size must be >= 1");
    let indices: Vec<usize> = (0..batch_size)
        .map(|_| rng.gen_range(windows.len()))
        .collect();

    let workers = n_workers.max(1).min(batch_size);

    let results: Vec<(Vec<crate::tensor::Tensor>, f32)> = if workers == 1 {
        // Serial path. Used when n_workers == 1 (deterministic) or when
        // batch_size == 1 (parallelising one window has no benefit and
        // costs a thread spawn).
        indices
            .iter()
            .map(|&i| {
                let (input, target) = &windows[i];
                compute_grads_for_window(model, input, target)
            })
            .collect()
    } else {
        // Parallel path. Split the index list into `workers` chunks; each
        // worker thread sequentially processes its chunk. Spawning fewer
        // threads than batch_size (when n_workers < batch_size) batches up
        // multiple windows on each thread, which is the right tradeoff
        // when you want to cap concurrent CPU usage.
        std::thread::scope(|s| {
            let chunk_size = indices.len().div_ceil(workers);
            let handles: Vec<_> = indices
                .chunks(chunk_size)
                .map(|chunk| {
                    let chunk_indices: Vec<usize> = chunk.to_vec();
                    s.spawn(move || {
                        chunk_indices
                            .iter()
                            .map(|&i| {
                                let (input, target) = &windows[i];
                                compute_grads_for_window(model, input, target)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();

            let mut all_results = Vec::with_capacity(indices.len());
            for h in handles {
                all_results.extend(h.join().unwrap());
            }
            all_results
        })
    };

    // Average gradients across the batch. Start from the first result's
    // tensors (move out, no clone) and sum the rest into them.
    let n = results.len() as f32;
    let mut iter = results.into_iter();
    let (mut avg_grads, mut total_loss) = iter
        .next()
        .expect("compute_batched_grads: batch_size was zero");
    for (other_grads, other_loss) in iter {
        for (a, b) in avg_grads.iter_mut().zip(&other_grads) {
            for (av, bv) in a.data.iter_mut().zip(&b.data) {
                *av += bv;
            }
        }
        total_loss += other_loss;
    }
    for g in avg_grads.iter_mut() {
        for v in g.data.iter_mut() {
            *v /= n;
        }
    }
    (avg_grads, total_loss / n)
}

/// Phase 5.a: CUDA-backed batched gradient computation. Mirrors
/// `compute_batched_grads`'s signature (averaged grads + mean loss
/// over `batch_size` windows) but each per-window forward+backward
/// runs through `CudaModel::compute_grads_for_window_bitnet` instead
/// of the CPU `Var`-based autograd path.
///
/// Implementation notes:
/// - Takes the device-resident `CudaModel` from the caller. The
///   training loop builds it once and refreshes it in place with
///   `CudaModel::sync_from_cpu` after each AdamW update (issue #1),
///   so the per-step cost is pure H->D copies with zero new device
///   allocations.
/// - Runs windows serially - the GPU is the parallelism layer here.
///   Spawning multiple CPU threads to dispatch CUDA kernels would just
///   serialise on the default stream anyway.
/// - The same window-index sampler (`Lcg::gen_range`) is used as the
///   CPU path, so swapping `use_cuda_backward` between two runs
///   preserves training-data ordering for an apples-to-apples
///   comparison at the same seed.
#[cfg(feature = "cuda")]
fn compute_batched_grads_cuda_bitnet(
    cuda_model: &crate::cuda::CudaModel,
    windows: &[(Vec<usize>, Vec<usize>)],
    rng: &mut crate::data::Lcg,
    batch_size: usize,
) -> (Vec<crate::tensor::Tensor>, f32) {
    assert!(batch_size >= 1, "batch_size must be >= 1");

    // Sample all window indices up front so changing batch_size does
    // not change which windows the run sees at the same RNG state.
    let indices: Vec<usize> = (0..batch_size)
        .map(|_| rng.gen_range(windows.len()))
        .collect();

    // Average gradients across the batch. Start from the first
    // result's tensors (move out, no clone) and sum the rest into them.
    let first_idx = indices[0];
    let (input, target) = &windows[first_idx];
    let (mut avg_grads, mut total_loss) = cuda_model.compute_grads_for_window_bitnet(input, target);
    for &idx in &indices[1..] {
        let (input, target) = &windows[idx];
        let (other_grads, other_loss) = cuda_model.compute_grads_for_window_bitnet(input, target);
        for (a, b) in avg_grads.iter_mut().zip(&other_grads) {
            for (av, bv) in a.data.iter_mut().zip(&b.data) {
                *av += bv;
            }
        }
        total_loss += other_loss;
    }
    let n = batch_size as f32;
    for g in avg_grads.iter_mut() {
        for v in g.data.iter_mut() {
            *v /= n;
        }
    }
    (avg_grads, total_loss / n)
}

/// Mean cross-entropy and perplexity (`exp(mean_ce)`) over a deterministic,
/// evenly-spaced subset of `val_windows`. Public so callers can run a final
/// eval after training without re-implementing the loop.
///
/// The eval uses a stride pattern (`i * stride` for `i in 0..n_samples`) so
/// the same windows get picked at every eval point - val_ppl trends are
/// comparable across the run, not contaminated by which random subset was
/// drawn that time.
pub fn eval_val_perplexity(
    model: &crate::model::Model,
    val_windows: &[(Vec<usize>, Vec<usize>)],
    n_samples: usize,
) -> (f32, f32) {
    if val_windows.is_empty() || n_samples == 0 {
        return (f32::NAN, f32::NAN);
    }
    let n = n_samples.min(val_windows.len());
    let stride = (val_windows.len() / n).max(1);
    let mut total_ce = 0.0_f32;
    let mut count = 0_usize;
    for i in 0..n {
        let (input, target) = &val_windows[i * stride];
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);
        let logits = model.forward(&leaves, input);
        let loss = logits.cross_entropy(target);
        total_ce += loss.value().data[0];
        count += 1;
    }
    let mean_ce = total_ce / count as f32;
    (mean_ce, mean_ce.exp())
}

/// Train the full BitNet LM on either the embedded TINY_CORPUS or a corpus
/// loaded from disk. AdamW + global-L2 grad clip + cosine-with-warmup LR.
/// Each step samples a uniformly random training window. If `cfg.val_split`
/// is positive, the tail of the encoded id stream is held out for periodic
/// validation evaluation; held-out perplexity is printed during training
/// alongside `train_loss` / `anchor_loss` / `min_seen`.
///
/// Returns `(initial_loss, min_loss_seen, trained_model, vocab, optim_snapshot)`.
/// The optim snapshot is the state of the AdamW optimiser at the end of
/// training - feed it back through `TrainConfig::start_from_optim` on the
/// next run for a momentum-preserving resume.
fn train_bitnet_lm(
    cfg: TrainConfig,
) -> (
    f32,
    f32,
    crate::model::Model,
    crate::data::Vocab,
    crate::optim::OptimState,
) {
    use crate::data::{Lcg, TINY_CORPUS, Vocab, make_windows, read_corpus};
    use crate::model::Model;
    use crate::optim::{AdamW, clip_grad_norm_tensors, cosine_lr};

    let corpus_owned: String = match &cfg.corpus_path {
        Some(p) => match read_corpus(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "could not read corpus {:?}: {}; falling back to TINY_CORPUS",
                    p, e
                );
                TINY_CORPUS.to_string()
            }
        },
        None => TINY_CORPUS.to_string(),
    };

    let vocab = Vocab::from_text(&corpus_owned);
    let ids = vocab.encode(&corpus_owned);

    // If we're resuming, the loaded model's config wins (vocab + dimensions
    // must match the loaded weights). Otherwise build a fresh model from the
    // requested config + corpus vocab.
    let (mut model, model_cfg) = if let Some(loaded) = cfg.start_from {
        let lc = loaded.config;
        if lc.vocab_size != vocab.size() {
            eprintln!(
                "warning: loaded model vocab_size {} differs from corpus vocab {}.\n  \
                 Token ids will not match between training-time and load-time vocabs;\n  \
                 generation will produce nonsense unless the same corpus was used originally.",
                lc.vocab_size,
                vocab.size()
            );
        }
        println!(
            "resuming from checkpoint: hidden {}, ffn {}, n_heads {}, head_dim {}, blocks {}, seq_len {}",
            lc.hidden_dim, lc.ffn_dim, lc.n_heads, lc.head_dim, lc.n_blocks, lc.max_seq_len
        );
        (loaded, lc)
    } else {
        let mut mc = cfg.model;
        mc.vocab_size = vocab.size();
        (Model::new(&mc, cfg.seed), mc)
    };

    // Train / validation split. The split point is taken from the END of
    // the id stream so all training windows (and thus all gradient updates)
    // come from corpus positions strictly before the validation tail. No
    // leakage of val tokens into training. With `val_split = 0.0` the val
    // slice is empty and validation is skipped entirely.
    let val_split_clamped = cfg.val_split.clamp(0.0, 0.5);
    let val_chars = ((ids.len() as f32) * val_split_clamped) as usize;
    let train_end = ids.len().saturating_sub(val_chars);
    let train_ids = &ids[..train_end];
    let val_ids = &ids[train_end..];

    let windows = make_windows(train_ids, model_cfg.max_seq_len);
    let val_windows = if val_ids.len() > model_cfg.max_seq_len + 1 {
        make_windows(val_ids, model_cfg.max_seq_len)
    } else {
        Vec::new()
    };

    let mut rng = Lcg::new(cfg.seed ^ 0xDEADBEEF);

    let mut opt = AdamW::new_for(&model, cfg.peak_lr);
    opt.beta1 = cfg.adamw_beta1;
    opt.beta2 = cfg.adamw_beta2;
    opt.weight_decay = cfg.weight_decay;
    if let Some(state) = cfg.start_from_optim {
        println!(
            "restoring AdamW state from checkpoint (step_count = {})",
            state.step_count
        );
        opt.restore(state);
    }

    println!(
        "── M9: BitNet LM training ──\n\
         corpus       = {} chars\n\
         vocab        = {}\n\
         model        = hidden {}, ffn {}, n_heads {}, head_dim {}, blocks {}, seq_len {}\n\
         train wins   = {}\n\
         val wins     = {} (split = {:.0}%)\n\
         batching     = batch_size {}, n_workers {}\n\
         optimiser    = AdamW(b1={}, b2={}, wd={}), peak_lr={:.1e}, floor_lr={:.1e},\n\
                        warmup={} steps, grad_clip={}, total_steps={} (offset {}, cosine_total {})",
        corpus_owned.chars().count(),
        vocab.size(),
        model_cfg.hidden_dim,
        model_cfg.ffn_dim,
        model_cfg.n_heads,
        model_cfg.head_dim,
        model_cfg.n_blocks,
        model_cfg.max_seq_len,
        windows.len(),
        val_windows.len(),
        val_split_clamped * 100.0,
        cfg.batch_size,
        cfg.n_workers,
        cfg.adamw_beta1,
        cfg.adamw_beta2,
        cfg.weight_decay,
        cfg.peak_lr,
        cfg.floor_lr,
        cfg.warmup_steps,
        cfg.grad_clip,
        cfg.n_steps,
        cfg.start_step_offset,
        cfg.cosine_total_steps.unwrap_or(cfg.n_steps),
    );

    let eval_loss = |m: &Model, input: &[usize], target: &[usize]| -> f32 {
        let tape = Tape::new();
        let leaves = m.register_leaves(&tape);
        let logits = m.forward(&leaves, input);
        logits.cross_entropy(target).value().data[0]
    };

    let (input0, target0) = (windows[0].0.clone(), windows[0].1.clone());
    let initial_loss = eval_loss(&model, &input0, &target0);
    let mut min_loss = initial_loss;

    let val_enabled = !val_windows.is_empty() && cfg.eval_every > 0 && cfg.val_eval_samples > 0;

    let batch_size = cfg.batch_size.max(1);
    let n_workers = cfg.n_workers.max(1);

    // Issue #1: the device-resident model is built once, then refreshed
    // in place each step via `sync_from_cpu` - zero per-step device
    // allocations. Lazily initialised so CPU-only runs never touch CUDA.
    #[cfg(feature = "cuda")]
    let mut cuda_model: Option<crate::cuda::CudaModel> = None;

    for step in 0..cfg.n_steps {
        // Sample `batch_size` random windows every step. With batch_size = 1
        // (the default for the tiny demo) this is exactly the prior single-
        // window path, just with one extra Vec allocation. With batch_size > 1
        // the per-window forward+backward passes parallelise across up to
        // `n_workers` threads, returning averaged gradients.
        // Continue the cosine schedule across resumes: pass `step + offset`
        // and the (offset + n_steps) total so a cumulative training session
        // walks one smooth cosine instead of repeating warmup-and-decay
        // every `cargo run`. With offset = 0 (fresh runs) the math reduces
        // to the v0.7-v0.13 schedule exactly.
        let total = cfg.cosine_total_steps.unwrap_or(cfg.n_steps);
        let lr = cosine_lr(
            step + cfg.start_step_offset,
            cfg.warmup_steps,
            total,
            cfg.peak_lr,
            cfg.floor_lr,
        );
        opt.lr = lr;

        let (mut grads, batch_loss) = if cfg.use_cuda_backward {
            #[cfg(feature = "cuda")]
            {
                let cm = match cuda_model.as_mut() {
                    Some(cm) => {
                        // Masters moved last step (AdamW); refresh the
                        // existing device buffers in place.
                        cm.sync_from_cpu(&model);
                        cm
                    }
                    None => cuda_model.insert(crate::cuda::CudaModel::from_cpu(&model)),
                };
                compute_batched_grads_cuda_bitnet(cm, &windows, &mut rng, batch_size)
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!(
                    "error: TrainConfig.use_cuda_backward = true but the binary was built\n       \
                     without the `cuda` feature. Rebuild with: cargo build --release --features cuda"
                );
                std::process::exit(2);
            }
        } else {
            compute_batched_grads(&model, &windows, &mut rng, batch_size, n_workers)
        };
        if batch_loss < min_loss {
            min_loss = batch_loss;
        }
        let pre_clip = clip_grad_norm_tensors(&mut grads, cfg.grad_clip);
        opt.step_with_grads(&mut model, &grads);

        let on_log_step = step % cfg.log_every == 0 || step == cfg.n_steps - 1;
        let on_eval_step = val_enabled && (step % cfg.eval_every == 0 || step == cfg.n_steps - 1);

        if on_log_step {
            let anchor_loss = eval_loss(&model, &input0, &target0);
            if on_eval_step {
                let (val_loss, val_ppl) =
                    eval_val_perplexity(&model, &val_windows, cfg.val_eval_samples);
                println!(
                    "step {:>5}   train_loss = {:.4}   anchor_loss = {:.4}   \
                     min_seen = {:.4}   val_loss = {:.4}   val_ppl = {:.2}   \
                     lr = {:.4e}   |g| = {:.3}",
                    step, batch_loss, anchor_loss, min_loss, val_loss, val_ppl, lr, pre_clip
                );
            } else {
                println!(
                    "step {:>5}   train_loss = {:.4}   anchor_loss = {:.4}   \
                     min_seen = {:.4}   lr = {:.4e}   |g| = {:.3}",
                    step, batch_loss, anchor_loss, min_loss, lr, pre_clip
                );
            }
        } else if on_eval_step {
            // Eval-only step (no training-loss log fires). Print a single
            // val_ppl line so the cadence stays visible.
            let (val_loss, val_ppl) =
                eval_val_perplexity(&model, &val_windows, cfg.val_eval_samples);
            println!(
                "step {:>5}   val_loss = {:.4}   val_ppl = {:.2}",
                step, val_loss, val_ppl
            );
        }
    }

    // Final, more accurate validation pass at end of training. Use 5× the
    // running-eval sample count so the headline number reflects the model
    // less noisily than any individual training-time eval.
    if val_enabled {
        let n = (cfg.val_eval_samples * 5).min(val_windows.len());
        let (val_loss, val_ppl) = eval_val_perplexity(&model, &val_windows, n);
        let baseline_ppl = vocab.size() as f32;
        println!(
            "\nfinal validation:  val_loss = {:.4}   val_ppl = {:.3}   \
             (uniform-vocab baseline = {:.1}, ratio = {:.3})",
            val_loss,
            val_ppl,
            baseline_ppl,
            val_ppl / baseline_ppl
        );
    }

    let optim_snapshot = opt.snapshot();
    (initial_loss, min_loss, model, vocab, optim_snapshot)
}

/// Directory where every trained model gets written. Created on demand by
/// `models_path` so the user never has to make it manually.
const MODELS_DIR: &str = "models";

/// Build a path under `models/` and ensure the directory exists. Failures to
/// create the directory are logged and the path is returned anyway, letting
/// the eventual `fs::write` produce its own clearer error.
fn models_path(filename: &str) -> std::path::PathBuf {
    if let Err(e) = std::fs::create_dir_all(MODELS_DIR) {
        eprintln!("could not create {}/: {}", MODELS_DIR, e);
    }
    std::path::PathBuf::from(MODELS_DIR).join(filename)
}

/// Entry point for the standalone Shakespeare training run. Called by
/// `cargo run --release -- shakespeare [resume_path]`. Trains (or continues
/// training) a larger BitNet on the full TinyShakespeare corpus, exports all
/// three formats, and generates samples in greedy + two temperature modes.
fn run_shakespeare_training(resume_path: Option<std::path::PathBuf>, large: bool, use_cuda: bool) {
    let mut cfg = if large {
        TrainConfig::shakespeare_large()
    } else {
        TrainConfig::shakespeare()
    };
    cfg.use_cuda_backward = use_cuda;
    if use_cuda {
        // GPU is the parallelism layer; running multiple threads to
        // dispatch CUDA kernels would just serialise on the default
        // stream. Force serial-batched mode.
        cfg.n_workers = 1;
    }
    let path_present = cfg.corpus_path.as_ref().is_some_and(|p| p.exists());
    if !path_present {
        eprintln!(
            "Could not find data/tinyshakespeare.txt.\n\
             Download it with:\n  \
             curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \\\n    \
             -o data/tinyshakespeare.txt"
        );
        std::process::exit(1);
    }

    // If a resume path is provided, load the model and seed the config with it.
    if let Some(p) = resume_path {
        let mut f = match std::fs::File::open(&p) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open resume file {}: {}", p.display(), e);
                std::process::exit(1);
            }
        };
        match export::import(&mut f) {
            Ok((m, fmt, optim)) => {
                let optim_msg = if optim.is_some() {
                    "with optim state"
                } else {
                    "no optim state (will reset AdamW momentum)"
                };
                println!(
                    "loaded {:?}-format checkpoint from {} ({})",
                    fmt,
                    p.display(),
                    optim_msg
                );
                cfg.start_from = Some(m);
                // Continue the cosine LR schedule from where the previous
                // run left off, instead of restarting at warmup-zero.
                // Diagnosed in the v0.13 cumulative-30k run: warmup-restart
                // perturbs converged 5M-param weights and consumed ~2k of
                // each 10k-step resume's productive budget. Reading the
                // step_count from the OPTM payload makes the new schedule
                // pick up at the right position automatically.
                if let Some(state) = optim.as_ref() {
                    cfg.start_step_offset = state.step_count as usize;
                    cfg.cosine_total_steps = Some(cfg.start_step_offset + cfg.n_steps);
                    println!(
                        "continuing cosine LR schedule: offset {} -> total {}",
                        cfg.start_step_offset,
                        cfg.cosine_total_steps.unwrap()
                    );
                }
                cfg.start_from_optim = optim;
            }
            Err(e) => {
                eprintln!("could not parse resume file {}: {}", p.display(), e);
                std::process::exit(1);
            }
        }
    }

    let (initial, min_seen, model, vocab, optim_state) = train_bitnet_lm(cfg);
    println!(
        "\nShakespeare training done.  initial = {:.4}   min seen = {:.4}   ratio = {:.2}",
        initial,
        min_seen,
        min_seen / initial
    );

    print_generation_samples(&model, &vocab, DEFAULT_PROMPTS);

    // Two artefacts side by side, each carrying only what its role needs:
    //
    //   - `shakespeare.f32.bin`  full-precision masters + AdamW optim state.
    //     A save+load cycle is a true identity: resuming from this path picks
    //     up at the same val_ppl the run ended on. This is the file
    //     `cargo run --release -- shakespeare <path>` should consume for
    //     clean continuations.
    //
    //   - `shakespeare.ternary_packed.bin`  the compact 1.58-bit-per-weight
    //     deployment artefact. Optim state is *not* included here: the AdamW
    //     `m`/`v` buffers are f32, and at the v0.9 ~2M-param scale they
    //     dominate the file (~8 MB vs ~200 KB of packed weights). The packed
    //     file is meant for distribution / inference, where momentum is
    //     irrelevant; resume always uses the .f32.bin instead.
    let mut f32_buf = Vec::new();
    export::export_f32(&model, &mut f32_buf, Some(&optim_state))
        .expect("f32 export to Vec cannot fail");
    let f32_path = models_path("shakespeare.f32.bin");
    let _ = std::fs::write(&f32_path, &f32_buf);
    println!(
        "\nwrote {} ({:.2} KB)  resume from this for byte-identical continuation",
        f32_path.display(),
        f32_buf.len() as f32 / 1024.0
    );

    let mut packed = Vec::new();
    export::export_ternary_packed(&model, &mut packed, None)
        .expect("packed export to Vec cannot fail");
    let path = models_path("shakespeare.ternary_packed.bin");
    let _ = std::fs::write(&path, &packed);
    println!(
        "wrote {} ({:.2} KB)  compact deployment artefact (no optim state; resume from .f32.bin)",
        path.display(),
        packed.len() as f32 / 1024.0
    );
}

/// Print the same five generation passes the trainer prints at the end
/// of a run, over the supplied prompt list: greedy, temperature, top-k,
/// top-p (T=0.8), top-p (T=0.5, v0.17), and KV-cache top-p. Used both by
/// `run_shakespeare_training` (post-train tail, three default prompts)
/// and the standalone `sample` subcommand below (caller-supplied prompt
/// when given, otherwise the same defaults).
fn print_generation_samples(model: &crate::model::Model, vocab: &data::Vocab, prompts: &[&str]) {
    use SampleMode::*;
    let modes = enabled_sample_modes();
    if modes.is_empty() {
        println!("\n(generation tail skipped: BITNET_SAMPLE_MODES=none)");
        return;
    }
    // One-line header so the user can see exactly which modes are
    // about to run (and infer the env-var spelling that selected them).
    let mode_labels: Vec<&str> = modes
        .iter()
        .map(|m| match m {
            Greedy => "greedy",
            Temp08 => "temp",
            TopK => "topk",
            TopP => "topp",
            TopP05 => "topp_low",
            KvCache => "kv",
        })
        .collect();
    println!("\n(sampling modes enabled: {})", mode_labels.join(","));

    if modes.contains(&Greedy) {
        println!("\n-- greedy generation --");
        for prompt in prompts {
            let g = inference::generate(model, vocab, prompt, 200);
            println!("\nprompt: {:?}\n{}", prompt, g);
        }
    }

    // Sampling modes: temperature alone (raw distribution shaped), top-k
    // (capped candidate set), top-p / nucleus (adaptive cumulative-probability
    // cutoff). Each call uses the same RNG so re-runs can be compared
    // bit-for-bit. Initialised once whether or not greedy ran above
    // (greedy is RNG-free, so its presence doesn't affect the stream).
    let mut rng = data::Lcg::new(0xCAFEF00D);

    if modes.contains(&Temp08) {
        println!("\n-- temperature sampling (T=0.8) --");
        for prompt in prompts {
            let g = inference::generate_with_mode(
                model,
                vocab,
                prompt,
                200,
                inference::SamplingMode::Temperature { temperature: 0.8 },
                &mut rng,
            );
            println!("\nprompt: {:?}\n{}", prompt, g);
        }
    }

    if modes.contains(&TopK) {
        println!("\n-- top-k sampling (k=10, T=0.8) --");
        for prompt in prompts {
            let g = inference::generate_with_mode(
                model,
                vocab,
                prompt,
                200,
                inference::SamplingMode::TopK {
                    k: 10,
                    temperature: 0.8,
                },
                &mut rng,
            );
            println!("\nprompt: {:?}\n{}", prompt, g);
        }
    }

    if modes.contains(&TopP) {
        println!("\n-- top-p / nucleus sampling (p=0.9, T=0.8) --");
        for prompt in prompts {
            let g = inference::generate_with_mode(
                model,
                vocab,
                prompt,
                200,
                inference::SamplingMode::TopP {
                    p: 0.9,
                    temperature: 0.8,
                },
                &mut rng,
            );
            println!("\nprompt: {:?}\n{}", prompt, g);
        }
    }

    // Lower-temperature top-p pass. Tightens the distribution to favour
    // the model's high-confidence completions; tends to produce more
    // syntactically grammatical output at the cost of variety. At
    // small char-LM scales the model has learnt local Shakespeare
    // patterns much more reliably than long-range coherence, so a
    // lower T often surfaces the cleaner short clauses without the
    // hallucinated words that T=0.8 produces. This is the highest-
    // signal single mode for "is this a working LM?" inspections.
    if modes.contains(&TopP05) {
        println!("\n-- top-p / nucleus sampling (p=0.9, T=0.5) --");
        for prompt in prompts {
            let g = inference::generate_with_mode(
                model,
                vocab,
                prompt,
                200,
                inference::SamplingMode::TopP {
                    p: 0.9,
                    temperature: 0.5,
                },
                &mut rng,
            );
            println!("\nprompt: {:?}\n{}", prompt, g);
        }
    }

    // KV-cache top-p (v0.16; sliding-window since v0.16.1). Same
    // architecture, same sampling mode, but the forward path keeps a
    // per-head cache of (K, V) rows. K is stored unrotated; RoPE is
    // applied at attention-score time using each row's logical (in-cache)
    // index. Cache caps at `max_seq_len` rows and slides (oldest evicted)
    // so positions stay in the trained range for arbitrarily long
    // generations. Per-step compute drops from O(t * H^2 * blocks) to
    // O(H^2 * blocks) - a ~50-100x wall-clock speedup. Output diverges
    // slightly from the Var path due to f32 summation order in attention.
    if modes.contains(&KvCache) {
        println!("\n-- KV-cache top-p (p=0.9, T=0.8; v0.16.1 sliding fast path) --");
        let kv_t0 = std::time::Instant::now();
        for prompt in prompts {
            let g = inference_kv::generate_with_cache(
                model,
                vocab,
                prompt,
                200,
                inference::SamplingMode::TopP {
                    p: 0.9,
                    temperature: 0.8,
                },
                &mut rng,
            );
            println!("\nprompt: {:?}\n{}", prompt, g);
        }
        println!(
            "\n(KV-cache top-p generated {} x 200 tokens in {:.2}s)",
            prompts.len(),
            kv_t0.elapsed().as_secs_f32()
        );
    }
}

/// Default prompt suite for the post-training generation tail and
/// the no-arg `sample` subcommand. Three short Shakespeare-shaped
/// stubs that exercise different opening contexts: a speaker tag,
/// a famous line opener, and a regal vocative.
const DEFAULT_PROMPTS: &[&str] = &["ROMEO:", "To be ", "King "];

/// Identifier for one of the six sampling modes the post-training tail
/// can run. The `print_generation_samples` helper consults
/// `enabled_sample_modes` at the start of each section to decide whether
/// to skip it. Order matters for output: modes always print in the
/// canonical order (Greedy, Temp08, TopK, TopP, TopP05, KvCache),
/// regardless of how the user lists them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SampleMode {
    Greedy,
    Temp08,
    TopK,
    TopP,
    TopP05,
    KvCache,
}

/// Read `BITNET_SAMPLE_MODES` and return the list of sampling passes the
/// post-training tail should run. The env var is comma-separated; mode
/// tokens are matched case-insensitively against a small whitelist of
/// aliases. Recognised values:
///
/// - `all` (default if unset) - run every mode in the canonical order.
/// - `min` / `minimal` - run only the two highest-signal modes:
///   top-p at T=0.5 (cleanest dialogue) and the KV-cache top-p
///   (different code path; smoke-tests the sliding fix).
/// - `none` - skip the generation tail entirely.
/// - Comma-separated tokens, any subset of:
///   `greedy`, `temp`, `topk`, `topp`, `topp_low` (T=0.5), `kv`.
///
/// Examples:
///   `BITNET_SAMPLE_MODES=min cargo run --release -- sample <path>`
///   `BITNET_SAMPLE_MODES=topp_low,kv cargo run -- shakespeare`
///   `BITNET_SAMPLE_MODES=greedy,topp cargo run -- sample <path> "ROMEO:"`
fn enabled_sample_modes() -> Vec<SampleMode> {
    use SampleMode::*;
    let raw = std::env::var("BITNET_SAMPLE_MODES").unwrap_or_default();
    let raw = raw.trim().to_ascii_lowercase();
    if raw.is_empty() || raw == "all" {
        return vec![Greedy, Temp08, TopK, TopP, TopP05, KvCache];
    }
    if raw == "min" || raw == "minimal" {
        return vec![TopP05, KvCache];
    }
    if raw == "none" {
        return Vec::new();
    }
    let mut requested: Vec<SampleMode> = Vec::new();
    for tok in raw.split(',') {
        let mode = match tok.trim() {
            "greedy" => Greedy,
            "temp" | "temperature" => Temp08,
            "topk" | "top-k" => TopK,
            "topp" | "top-p" => TopP,
            "topp_low" | "topp-low" | "topp-cold" | "topp05" => TopP05,
            "kv" | "kv_cache" | "kv-cache" | "kvcache" => KvCache,
            "" => continue,
            other => {
                eprintln!(
                    "warning: BITNET_SAMPLE_MODES contains unknown mode {:?}; ignored. \
                     Recognised: greedy, temp, topk, topp, topp_low, kv (or `all` / `min` / `none`).",
                    other
                );
                continue;
            }
        };
        if !requested.contains(&mode) {
            requested.push(mode);
        }
    }
    // Force canonical print order regardless of how the user listed them.
    let canonical = [Greedy, Temp08, TopK, TopP, TopP05, KvCache];
    canonical
        .iter()
        .copied()
        .filter(|m| requested.contains(m))
        .collect()
}

/// Standalone "sample only" entry point. Loads a checkpoint
/// (`.f32.bin` or `.ternary*.bin`), reconstructs the vocab from the
/// training corpus on disk, and prints the same five generation passes
/// the trainer prints. Lets the user verify a model's generation quality
/// without paying the ~50-minute cost of a fresh training run; especially
/// useful for verifying inference-path bug fixes (e.g. the v0.16.1
/// KV-cache sliding-window fix) against an existing checkpoint.
fn run_sample_only(path: std::path::PathBuf) {
    use crate::data::{Vocab, read_corpus};

    if !path.exists() {
        eprintln!("checkpoint not found: {}", path.display());
        std::process::exit(1);
    }
    let corpus_path = std::path::PathBuf::from("data/tinyshakespeare.txt");
    if !corpus_path.exists() {
        eprintln!(
            "Could not find {} (needed to rebuild the same vocab the model was trained against).\n\
             Download it with:\n  \
             curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \\\n    \
             -o data/tinyshakespeare.txt",
            corpus_path.display()
        );
        std::process::exit(1);
    }

    let mut f = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open checkpoint {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };
    let (model, fmt, _optim) = match export::import(&mut f) {
        Ok(triple) => triple,
        Err(e) => {
            eprintln!("could not parse checkpoint {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };
    println!("loaded {:?}-format checkpoint from {}", fmt, path.display());
    println!(
        "model        = hidden {}, ffn {}, n_heads {}, head_dim {}, blocks {}, seq_len {}",
        model.config.hidden_dim,
        model.config.ffn_dim,
        model.config.n_heads,
        model.config.head_dim,
        model.config.n_blocks,
        model.config.max_seq_len
    );

    let corpus = match read_corpus(&corpus_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not read {}: {}", corpus_path.display(), e);
            std::process::exit(1);
        }
    };
    let vocab = Vocab::from_text(&corpus);
    if vocab.size() != model.config.vocab_size {
        eprintln!(
            "vocab size mismatch: corpus has {} chars, checkpoint expects {}.\n\
             The corpus on disk has drifted from the one this model was trained on.",
            vocab.size(),
            model.config.vocab_size
        );
        std::process::exit(1);
    }

    print_generation_samples(&model, &vocab, DEFAULT_PROMPTS);
}

/// Variant of `run_sample_only` that takes a caller-supplied prompt and
/// runs all six sampling modes on just that one prompt. The prompt is
/// silently filtered to the trained vocab (out-of-vocab characters get
/// dropped instead of panicking, so "feed it random BS" stays friendly).
fn run_sample_only_with_prompt(path: std::path::PathBuf, raw_prompt: String) {
    use crate::data::{Vocab, read_corpus};

    if !path.exists() {
        eprintln!("checkpoint not found: {}", path.display());
        std::process::exit(1);
    }
    let corpus_path = std::path::PathBuf::from("data/tinyshakespeare.txt");
    if !corpus_path.exists() {
        eprintln!(
            "Could not find {} (needed to rebuild the same vocab the model was trained against).",
            corpus_path.display()
        );
        std::process::exit(1);
    }

    let mut f = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open checkpoint {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };
    let (model, fmt, _optim) = match export::import(&mut f) {
        Ok(triple) => triple,
        Err(e) => {
            eprintln!("could not parse checkpoint {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };
    println!("loaded {:?}-format checkpoint from {}", fmt, path.display());

    let corpus = match read_corpus(&corpus_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not read {}: {}", corpus_path.display(), e);
            std::process::exit(1);
        }
    };
    let vocab = Vocab::from_text(&corpus);
    if vocab.size() != model.config.vocab_size {
        eprintln!(
            "vocab size mismatch: corpus has {} chars, checkpoint expects {}.\n\
             The corpus on disk has drifted from the one this model was trained on.",
            vocab.size(),
            model.config.vocab_size
        );
        std::process::exit(1);
    }

    // Filter prompt to chars present in the trained vocab; report what
    // got dropped (if anything) so the user knows. Vocab is the 65 chars
    // of TinyShakespeare - mostly printable ASCII letters / digits /
    // common punctuation / newline / space - so most natural English
    // prompts pass through unchanged.
    let in_vocab: std::collections::HashSet<char> = corpus.chars().collect();
    let filtered: String = raw_prompt
        .chars()
        .filter(|c| in_vocab.contains(c))
        .collect();
    let dropped: String = raw_prompt
        .chars()
        .filter(|c| !in_vocab.contains(c))
        .collect();
    if !dropped.is_empty() {
        eprintln!(
            "warning: dropped {} out-of-vocab char(s) from prompt: {:?}",
            dropped.chars().count(),
            dropped
        );
    }
    if filtered.is_empty() {
        eprintln!("prompt empty after vocab filter; nothing to feed the model");
        std::process::exit(1);
    }
    println!(
        "prompt (in-vocab) = {:?}  ({} chars)",
        filtered,
        filtered.chars().count()
    );

    print_generation_samples(&model, &vocab, &[filtered.as_str()]);
}

/// End-to-end CPU vs GPU model forward benchmark. Builds a tiny model
/// (config matches the `cuda::tests` Phase 3 case), runs N forwards on
/// each backend, reports mean latency. Demonstrates the full Phase 3
/// pipeline (embed -> blocks -> rmsnorm -> lm_head matmul) running
/// device-resident on the GPU.
#[cfg(feature = "cuda")]
fn run_cuda_forward_bench() {
    use std::time::Instant;

    use crate::cuda::{CudaModel, cuda_state};
    use crate::device::{FfnWeights, HeadWeights, block_inference};
    use crate::model::{Model, ModelConfig};
    use crate::tensor::Tensor;

    println!("── CUDA Phase 3: end-to-end model forward (CPU vs GPU) ──");

    let t_init = Instant::now();
    if cuda_state().is_err() {
        eprintln!("CUDA initialisation failed");
        return;
    }
    println!(
        "device init + cuBLAS handle + NVRTC compile: {:.2} ms",
        t_init.elapsed().as_secs_f64() * 1e3
    );

    // Two configs: one matching the test case (tiny, ~10k params) and
    // one closer to v0.13 shakespeare (larger, ~250k params; still
    // smaller than the trained checkpoint to keep the demo fast).
    let configs: &[(&str, ModelConfig)] = &[
        (
            "tiny  (vocab 17, hidden 32, 4 heads, 2 blocks)",
            ModelConfig {
                vocab_size: 17,
                hidden_dim: 32,
                n_heads: 4,
                head_dim: 8,
                ffn_dim: 64,
                max_seq_len: 16,
                n_blocks: 2,
            },
        ),
        (
            "med   (vocab 65, hidden 96, 6 heads, 4 blocks)",
            ModelConfig {
                vocab_size: 65,
                hidden_dim: 96,
                n_heads: 6,
                head_dim: 16,
                ffn_dim: 192,
                max_seq_len: 32,
                n_blocks: 4,
            },
        ),
    ];

    let iters = 20usize;
    println!(
        "\n{:<55}  {:>10}  {:>10}  {:>10}",
        "config", "cpu (us)", "cuda (us)", "ratio"
    );
    println!("{:-<55}  {:->10}  {:->10}  {:->10}", "", "", "", "");

    for (label, config) in configs {
        let model = Model::new(config, 1337);
        let ids: Vec<usize> = (0..config.max_seq_len.min(8))
            .map(|i| i % config.vocab_size)
            .collect();

        // CPU forward via the trait surface (no autograd, no quant) -
        // matches what the GPU forward computes mathematically.
        let cpu_forward = |model: &Model, ids: &[usize]| -> Tensor {
            let h = model.config.hidden_dim;
            let table = &model.token_embed.data;
            let mut slab: Vec<f32> = Vec::with_capacity(ids.len() * h);
            for &id in ids {
                slab.extend_from_slice(&table[id * h..(id + 1) * h]);
            }
            let mut x = Tensor::from_vec(slab, vec![ids.len(), h]);
            for b in &model.blocks {
                let heads: Vec<HeadWeights<Tensor>> = b
                    .heads
                    .iter()
                    .map(|h| HeadWeights {
                        w_q: h.w_q.clone(),
                        w_k: h.w_k.clone(),
                        w_v: h.w_v.clone(),
                        w_o: h.w_o.clone(),
                    })
                    .collect();
                let ffn = FfnWeights {
                    w_gate: b.ffn_gate_w.clone(),
                    w_up: b.ffn_up_w.clone(),
                    w_down: b.ffn_down_w.clone(),
                };
                x = block_inference(&x, &heads, &ffn, model.config.head_dim);
            }
            // Tied LM head: same tensor as token_embed, transposed.
            x.rmsnorm().matmul(&model.token_embed.transpose_2d())
        };

        // Build the GPU model once (weights stay resident across all
        // iterations - this is the core Phase 3 advantage over Phase 1's
        // per-call H<->D copy).
        let cuda_model = CudaModel::from_cpu(&model);

        // Warmup.
        let _ = cpu_forward(&model, &ids);
        let _ = cuda_model.forward(&ids).to_cpu().expect("D->H failed");

        let cpu_start = Instant::now();
        for _ in 0..iters {
            let _ = cpu_forward(&model, &ids);
        }
        let cpu_us = cpu_start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let cuda_start = Instant::now();
        for _ in 0..iters {
            let _ = cuda_model.forward(&ids).to_cpu().expect("D->H failed");
        }
        let cuda_us = cuda_start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "{:<55}  {:>10.1}  {:>10.1}  {:>10.2}",
            label,
            cpu_us,
            cuda_us,
            cuda_us / cpu_us,
        );
    }

    println!("\nNote: at these tiny model sizes the GPU is far slower than the CPU.");
    println!("      The forward queues 60-80 kernel launches per call and each launch");
    println!("      pays ~10-30 us of fixed driver overhead, regardless of how cheap the");
    println!("      kernel itself is. v0.13 scale (~5M params, hidden 192) gives each");
    println!("      kernel real work to do and should flip the ratio. Per-call sync was");
    println!("      already stripped from the trait impls; the remaining wins live in");
    println!("      kernel fusion, CUDA graphs (capture once, replay many), batched");
    println!("      forwards, and Phase 5's ternary tensor-core GEMM.");
}

/// CPU-vs-CUDA matmul demo + benchmark. Reports per-call latency for the
/// three matmul shapes the v0.13 model leans on hardest, alongside the
/// max absolute error between the CPU and GPU paths so the bit-equality
/// caveat is visible in the output (and not just in docs).
///
/// The GPU column includes the H<->D copies. At v0.13 model scale that
/// dominates kernel time (a 192x192 @ 192x16 matmul is ~6 us of compute
/// on a 4070 but ~50-100 us of PCIe round-trip), which is the headline
/// argument for Phase 2: keep activations + weights resident on device
/// across a whole forward pass instead of round-tripping per matmul.
#[cfg(feature = "cuda")]
fn run_cuda_demo() {
    use std::time::Instant;

    use crate::cuda::{cuda_matmul, cuda_state};
    use crate::data::Lcg;
    use crate::tensor::Tensor;

    println!("── CUDA matmul: CPU vs cuBLAS sgemm (Chunk 2.0) ──");

    // Initialise device + compile NVRTC kernel up front so the first
    // benchmarked call does not include the ~50 ms one-time compile.
    let t_init = Instant::now();
    match cuda_state() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("CUDA initialisation failed: {e}");
            return;
        }
    }
    println!(
        "device init + cuBLAS handle: {:.2} ms",
        t_init.elapsed().as_secs_f64() * 1e3
    );

    // Three representative shapes from v0.13's hot path. (m, k, n).
    let shapes: &[(usize, usize, usize, &str)] = &[
        (
            64,
            192,
            16,
            "attention Q  [seq 64, hidden 192] @ W_q  [192, 16]   = [64, 16]",
        ),
        (
            64,
            192,
            384,
            "FFN gate/up  [seq 64, hidden 192] @ W   [192, 384]  = [64, 384]",
        ),
        (
            64,
            384,
            192,
            "FFN down     [seq 64, ffn 384]  @ W_d   [384, 192]  = [64, 192]",
        ),
    ];

    let mut rng = Lcg::new(0x_C0FFEE_C001_u64);
    let iters = 100usize;

    println!(
        "\n{:<60}  {:>10}  {:>10}  {:>10}  {:>12}",
        "shape", "cpu (us)", "cuda (us)", "ratio", "max |diff|"
    );
    println!(
        "{:-<60}  {:->10}  {:->10}  {:->10}  {:->12}",
        "", "", "", "", ""
    );

    for &(m, k, n, label) in shapes {
        let lhs_data: Vec<f32> = (0..m * k).map(|_| rng.next_f01() - 0.5).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|_| rng.next_f01() - 0.5).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        // One warmup call per side so caches / first-touch allocations
        // do not skew the steady-state measurement.
        let _ = lhs.matmul(&rhs);
        let _ = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");

        let cpu_start = Instant::now();
        let mut cpu_out = lhs.matmul(&rhs);
        for _ in 1..iters {
            cpu_out = lhs.matmul(&rhs);
        }
        let cpu_us = cpu_start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let cuda_start = Instant::now();
        let mut cuda_out = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");
        for _ in 1..iters {
            cuda_out = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");
        }
        let cuda_us = cuda_start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let max_diff = cpu_out
            .data
            .iter()
            .zip(&cuda_out.data)
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        println!(
            "{:<60}  {:>10.1}  {:>10.1}  {:>10.2}  {:>12.3e}",
            label,
            cpu_us,
            cuda_us,
            cuda_us / cpu_us,
            max_diff,
        );
    }

    println!("\nNote: per-call cuda time includes two H->D copies and one D->H copy.");
    println!("      The cuBLAS sgemm kernel itself is ~5-10 us on a 4070; the rest is");
    println!("      PCIe round-trip + the per-call cuBLAS setup. Phase 2 will keep tensors");
    println!("      device-resident across whole blocks so the copy cost is paid once per");
    println!("      forward pass instead of once per matmul, which should push the");
    println!("      speedup from ~1.6-3x today into the 5-10x range at v0.13 scale.");
}

/// Phase 4 chunk 4.5.f: end-to-end GPU training proof-of-concept.
/// Builds a small model from random init, trains it for `n_steps`
/// using `CudaModel::compute_grads_for_window` for the forward+
/// backward pass, applies AdamW updates on CPU. Verifies that loss
/// decreases meaningfully so the entire Phase 4 stack is exercised
/// in a real training context.
///
/// **Important caveat**: this path is **f32 throughout** - no BitNet
/// ternary STE quantisation in either forward or backward. Trains an
/// f32 transformer, NOT a ternary BitNet. Phase 5 (ternary tensor
/// cores) restores BitNet semantics on the GPU. Until then, treat
/// this as a Phase 4 wiring proof, not a production training path.
///
/// Per-step cost analysis (v0.13-shape demo, 7940HS + RTX 4070 Laptop):
///   - refresh device weights from CPU masters: `sync_from_cpu`
///     overwrites the existing buffers in place via `memcpy_htod`
///     (issue #1), so the step pays pure H->D copy cost - allocation
///     happens once at startup. At v0.13 scale (~5M params, ~20 MB
///     f32) the PCIe Gen4 copy is well under 1 ms.
///   - forward+backward through one block stack: tens of kernel
///     launches at ~10-30 us each (mostly launch overhead, not
///     kernel time). Several ms per window at v0.13 scale.
///   - optimiser update on CPU: linear in param count, ~few ms.
#[cfg(feature = "cuda")]
fn run_cuda_train_demo() {
    use crate::cuda::{CudaModel, cuda_state};
    use crate::data::{Lcg, TINY_CORPUS, Vocab, make_windows};
    use crate::model::{Model, ModelConfig};
    use crate::optim::{AdamW, clip_grad_norm_tensors};
    use std::time::Instant;

    if cuda_state().is_err() {
        eprintln!(
            "error: cuda_state() failed - is the CUDA runtime available? \
             try: PATH=/opt/cuda/bin:$PATH CUDA_PATH=/opt/cuda \\\n             \
             cargo run --release --features cuda -- cuda-train-demo"
        );
        std::process::exit(2);
    }
    println!("── cuda-train-demo: end-to-end GPU forward+backward + CPU optimiser ──");
    println!("    (Phase 4 chunk 4.5.f wiring proof - f32 throughout, no ternary STE)\n");

    // Tiny config so the demo finishes in a few seconds. Same vocab
    // size as TINY_CORPUS produces, small attention + FFN dims so
    // each window's GPU forward+backward is dominated by launch
    // overhead but still runs end-to-end.
    let vocab = Vocab::from_text(TINY_CORPUS);
    let model_cfg = ModelConfig {
        vocab_size: vocab.size(),
        hidden_dim: 32,
        n_heads: 4,
        head_dim: 8,
        ffn_dim: 64,
        max_seq_len: 16,
        n_blocks: 2,
    };
    let mut model = Model::new(&model_cfg, 1337);
    let n_steps = 100usize;
    let peak_lr = 5e-3_f32;

    let ids = vocab.encode(TINY_CORPUS);
    let windows = make_windows(&ids, model_cfg.max_seq_len);
    assert!(!windows.is_empty(), "TINY_CORPUS produced no windows");
    let mut rng = Lcg::new(1337);

    let mut opt = AdamW::new_for(&model, peak_lr);

    println!(
        "config: hidden={}  n_heads={}  ffn={}  blocks={}  seq={}  vocab={}",
        model_cfg.hidden_dim,
        model_cfg.n_heads,
        model_cfg.ffn_dim,
        model_cfg.n_blocks,
        model_cfg.max_seq_len,
        vocab.size(),
    );
    println!("running {n_steps} steps with peak_lr = {peak_lr:.4}\n");

    let mut min_loss: f32 = f32::INFINITY;
    let mut initial_loss: f32 = f32::NAN;
    let t0 = Instant::now();
    // Built once; refreshed in place each step (issue #1). The CPU
    // weights mutate after every opt step, so the device copy is
    // synced at the top of each iteration - zero per-step device
    // allocations.
    let mut cuda_model = CudaModel::from_cpu(&model);
    for step in 0..n_steps {
        let idx = rng.gen_range(windows.len());
        let (input, target) = &windows[idx];
        if step > 0 {
            cuda_model.sync_from_cpu(&model);
        }
        let (mut grads, loss) = cuda_model.compute_grads_for_window(input, target);
        if step == 0 {
            initial_loss = loss;
        }
        if loss < min_loss {
            min_loss = loss;
        }
        let _pre_clip = clip_grad_norm_tensors(&mut grads, 1.0);
        opt.step_with_grads(&mut model, &grads);
        if step % 10 == 0 || step == n_steps - 1 {
            println!(
                "step {:>4}   loss = {:.4}   min_seen = {:.4}",
                step, loss, min_loss
            );
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "\ndone: {n_steps} steps in {:.1}s  ({:.1} ms/step)",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() * 1000.0 / n_steps as f32,
    );
    println!(
        "  initial_loss = {:.4}   final-window_loss = (last printed line)   min_seen = {:.4}",
        initial_loss, min_loss,
    );
    if min_loss < initial_loss * 0.7 {
        println!(
            "  loss decreased {:.1}% from initial - Phase 4 GPU training stack works.",
            (1.0 - min_loss / initial_loss) * 100.0
        );
    } else {
        println!(
            "  loss decreased {:.1}% from initial. Consider longer run / different hyperparameters.",
            (1.0 - min_loss / initial_loss) * 100.0
        );
    }
}

fn main() {
    // CLI dispatch:
    //   cargo run --release                                       -- runs M4-M10 demo
    //   cargo run --release -- shakespeare                        -- fresh ~5M training
    //   cargo run --release -- shakespeare <resume_path>          -- resume ~5M
    //   cargo run --release -- shakespeare-large                  -- fresh ~8.5M training (seq 128)
    //   cargo run --release -- shakespeare-large <resume_path>    -- resume ~8.5M
    //   cargo run --release -- sample <checkpoint_path>           -- skip training; print samples on the 3 default prompts
    //   cargo run --release -- sample <checkpoint_path> <prompt..> -- skip training; print samples on a caller-supplied prompt
    //   cargo run --release --features cuda -- cuda-shakespeare           -- fresh ~5M GPU bitnet training
    //   cargo run --release --features cuda -- cuda-shakespeare <path>    -- resume ~5M on GPU
    //   cargo run --release --features cuda -- cuda-shakespeare-large     -- fresh ~8.5M GPU bitnet
    //   cargo run --release --features cuda -- cuda-demo          -- CPU vs CUDA matmul timings
    //   cargo run --release --features cuda -- cuda-forward-bench -- end-to-end CudaModel forward bench
    //   cargo run --release --features cuda -- cuda-train-demo    -- Phase 4 GPU forward+backward proof
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && (args[1] == "shakespeare" || args[1] == "shakespeare-large") {
        let large = args[1] == "shakespeare-large";
        let resume_path = args
            .get(2)
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists());
        run_shakespeare_training(resume_path, large, /*use_cuda=*/ false);
        return;
    }
    if args.len() > 1 && args[1] == "sample" {
        let path = match args.get(2) {
            Some(s) => std::path::PathBuf::from(s),
            None => {
                eprintln!(
                    "usage: cargo run --release -- sample <checkpoint_path> [prompt...]\n\
                     examples:\n  \
                     cargo run --release -- sample models/shakespeare.f32.bin\n  \
                     cargo run --release -- sample models/shakespeare.f32.bin \"BANANA:\"\n  \
                     cargo run --release -- sample models/shakespeare.f32.bin Look thee well"
                );
                std::process::exit(2);
            }
        };
        // Anything after the path is treated as the prompt. Joining with
        // single spaces lets the user skip shell-quoting for multi-word
        // prompts ("Look thee well" works, as does "Look\\ thee\\ well").
        if args.len() > 3 {
            let raw_prompt = args[3..].join(" ");
            run_sample_only_with_prompt(path, raw_prompt);
        } else {
            run_sample_only(path);
        }
        return;
    }
    if args.len() > 1 && (args[1] == "cuda-shakespeare" || args[1] == "cuda-shakespeare-large") {
        #[cfg(feature = "cuda")]
        {
            let large = args[1] == "cuda-shakespeare-large";
            let resume_path = args
                .get(2)
                .map(|s| std::path::PathBuf::from(s))
                .filter(|p| p.exists());
            run_shakespeare_training(resume_path, large, /*use_cuda=*/ true);
            return;
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "error: this binary was built without the `cuda` feature.\n       \
                 rebuild with: cargo build --release --features cuda"
            );
            std::process::exit(2);
        }
    }
    #[cfg(feature = "cuda")]
    if args.len() > 1 && args[1] == "cuda-demo" {
        run_cuda_demo();
        return;
    }
    #[cfg(feature = "cuda")]
    if args.len() > 1 && args[1] == "cuda-forward-bench" {
        run_cuda_forward_bench();
        return;
    }
    #[cfg(feature = "cuda")]
    if args.len() > 1 && args[1] == "cuda-train-demo" {
        run_cuda_train_demo();
        return;
    }
    #[cfg(not(feature = "cuda"))]
    if args.len() > 1
        && (args[1] == "cuda-demo"
            || args[1] == "cuda-forward-bench"
            || args[1] == "cuda-train-demo")
    {
        eprintln!("error: this binary was built without the `cuda` feature.");
        eprintln!("       rebuild with: cargo build --release --features cuda");
        std::process::exit(2);
    }

    println!("── M4: 1D linear regression by SGD on  y = 2x ──");
    let (m4_initial, m4_final, m4_w) = train_linear_regression(200, 0.05);
    println!(
        "\nM4 result:  initial loss = {:.6}   final loss = {:.6e}   w = {:.6} (target 2.0)\n",
        m4_initial, m4_final, m4_w
    );

    println!("── M6: 2D regression through BitLinear (STE) on  y = 2x₁ − 2x₂ ──");
    let (m6_initial, m6_final, m6_w) = train_bitlinear_regression(500, 0.05);
    println!(
        "\nM6 result:  initial loss = {:.6}   final loss = {:.6e}   w = [{:.4}, {:.4}] (target [2, -2])\n",
        m6_initial, m6_final, m6_w[0], m6_w[1]
    );

    let (m9_initial, m9_min, m9_model, m9_vocab, _m9_optim) =
        train_bitnet_lm(TrainConfig::tiny_demo());
    println!(
        "\nM9 result:  initial loss = {:.4}   min loss seen = {:.4}   ratio = {:.2}",
        m9_initial,
        m9_min,
        m9_min / m9_initial
    );

    // ── M10: greedy inference on the trained model. ──
    println!("\n── M10: greedy generation from the trained BitNet LM ──");
    let prompts = ["to be ", "the ", "or "];
    for prompt in &prompts {
        let generated = inference::generate(&m9_model, &m9_vocab, prompt, 40);
        // {:?} so we see literal newlines / whitespace clearly.
        println!("prompt {:>10?}  →  {:?}", prompt, generated);
    }

    // ── M10: ternary export - the on-disk payoff ──
    println!("\n── M10: ternary export size comparison ──");

    let mut f32_buf = Vec::new();
    let mut ternary_buf = Vec::new();
    let mut packed_buf = Vec::new();
    let f32_size =
        export::export_f32(&m9_model, &mut f32_buf, None).expect("f32 export to Vec cannot fail");
    let ternary_size = export::export_ternary(&m9_model, &mut ternary_buf, None)
        .expect("ternary export to Vec cannot fail");
    let packed_size = export::export_ternary_packed(&m9_model, &mut packed_buf, None)
        .expect("packed export to Vec cannot fail");

    let kb = |n: usize| n as f32 / 1024.0;
    println!(
        "f32 export:           {:>6} bytes ({:.2} KB)",
        f32_size,
        kb(f32_size)
    );
    println!(
        "ternary (i8/value):   {:>6} bytes ({:.2} KB)   {:.2}x vs f32",
        ternary_size,
        kb(ternary_size),
        f32_size as f32 / ternary_size as f32
    );
    println!(
        "ternary (5-per-byte): {:>6} bytes ({:.2} KB)   {:.2}x vs f32",
        packed_size,
        kb(packed_size),
        f32_size as f32 / packed_size as f32
    );

    // Write all three so the user can inspect with `ls -la models/`.
    let demo_f32 = models_path("demo.f32.bin");
    let demo_ternary = models_path("demo.ternary.bin");
    let demo_packed = models_path("demo.ternary_packed.bin");
    let _ = std::fs::write(&demo_f32, &f32_buf);
    let _ = std::fs::write(&demo_ternary, &ternary_buf);
    let _ = std::fs::write(&demo_packed, &packed_buf);
    println!(
        "wrote {}, {}, {}",
        demo_f32.display(),
        demo_ternary.display(),
        demo_packed.display()
    );

    // Round-trip sanity check: load the packed file back and run greedy
    // generation on the same prompts. Output won't match exactly (the master
    // weights lost f32 precision), but it should still produce text.
    println!(
        "\nround-trip check: loading {} and generating...",
        demo_packed.display()
    );
    let mut cursor = std::io::Cursor::new(packed_buf);
    match export::import(&mut cursor) {
        Ok((loaded_model, fmt, _opt)) => {
            println!(
                "loaded {:?}-format model with vocab={}",
                fmt, loaded_model.config.vocab_size
            );
            let g = inference::generate(&loaded_model, &m9_vocab, "to be ", 30);
            println!("loaded -> {:?}", g);
        }
        Err(e) => println!("import failed: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE M4 GATE.
    /// If autograd + SGD work, loss must collapse, and w must converge to ≈ 2.0.
    /// If this test fails, every previous "passing" autograd test is a lie -
    /// some chain-rule wiring is wrong in a way that doesn't show up in
    /// closed-form single-op tests but does show up under iterative gradient descent.
    #[test]
    fn training_drives_loss_down_and_w_to_target() {
        // 200 steps at lr=0.05 is overkill (theoretical decay rate per step ≈ 0.56,
        // so error is ~10⁻⁵⁰ at the end), which makes the test extremely robust.
        let (initial, finalv, w) = train_linear_regression(200, 0.05);

        // 1. Loss must collapse. We assert at least 100x reduction; convergence is
        //    actually many orders of magnitude better, so the slack here is intentional.
        assert!(
            finalv < initial * 0.01,
            "loss did not drop enough: initial = {}, final = {}",
            initial,
            finalv
        );

        // 2. w must land near the true value 2.0.
        //    1e-3 tolerance is generous; expect to be at f32 precision in practice.
        assert!(
            (w - 2.0).abs() < 1e-3,
            "w did not converge: got {}, expected ≈ 2.0",
            w
        );
    }

    /// THE M6 GATE.
    /// Train through STE-quantised weights and STE-quantised activations.
    /// If STE works, master_w converges to ≈ (2, −2) - exactly representable
    /// on the ternary grid (γ=2, W_q=[+1, −1]). If any quantiser's backward
    /// returned 0 instead of identity, master_w would stay at (0, 0) forever
    /// and this test would catch it the same way the M4 test would catch a
    /// broken plain matmul backward.
    #[test]
    fn bitlinear_training_drives_loss_down_through_ste() {
        let (initial, final_, w) = train_bitlinear_regression(500, 0.05);

        // Loss must collapse. Theoretical floor ≈ 6e-5 (irreducible due to
        // INT8 grid bias on rows 2/3); 100× reduction from initial=4.0 is plenty.
        assert!(
            final_ < initial * 0.01,
            "BitLinear loss did not drop enough: initial = {}, final = {}",
            initial,
            final_
        );

        // Master weights must converge to the ternary-grid target (±2). Tolerance
        // is generous - actual convergence is to ≈ (2.008, −2.008) per the
        // back-of-envelope analysis. Anything within 0.5 proves STE is doing its job.
        assert!(
            (w[0] - 2.0).abs() < 0.5,
            "w[0] did not converge: got {}, expected ≈ +2.0",
            w[0]
        );
        assert!(
            (w[1] + 2.0).abs() < 0.5,
            "w[1] did not converge: got {}, expected ≈ −2.0",
            w[1]
        );
    }

    /// THE M9 GATE.
    /// Train the full BitNet LM on TINY_CORPUS for 300 steps.  At least one
    /// step must achieve loss < 0.7 × initial - that's "the architecture is
    /// trainable end-to-end on real text" demonstrated.
    /// We use min-loss-seen rather than last-step loss because STE training
    /// has noisy per-step loss (see model.rs `training_steps_reduce_loss_...`).
    #[test]
    fn bitnet_lm_training_drives_loss_down_substantially() {
        // Quieter config than the demo so the test runs faster.
        let mut cfg = TrainConfig::tiny_demo();
        cfg.n_steps = 200;
        cfg.log_every = usize::MAX; // suppress per-step prints inside the test
        let (initial, min_loss, _model, _vocab, _optim) = train_bitnet_lm(cfg);
        assert!(
            min_loss < initial * 0.7,
            "BitNet LM did not improve enough: initial = {}, min seen = {}, ratio = {:.3}",
            initial,
            min_loss,
            min_loss / initial
        );
    }

    /// `eval_val_perplexity` should produce finite, sensible values:
    /// `val_loss` non-negative finite, `val_ppl` between 1 and the
    /// uniform-vocab baseline (`vocab_size`) for a model that has done
    /// at least *some* training. Untrained-model perplexity sits very
    /// close to the baseline; trained perplexity drops below it.
    #[test]
    fn eval_val_perplexity_returns_sensible_values() {
        use crate::data::{TINY_CORPUS, Vocab, make_windows};
        use crate::model::{Model, ModelConfig};

        let vocab = Vocab::from_text(TINY_CORPUS);
        let ids = vocab.encode(TINY_CORPUS);
        let cfg = ModelConfig {
            vocab_size: vocab.size(),
            hidden_dim: 8,
            n_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            max_seq_len: 8,
            n_blocks: 1,
        };
        let model = Model::new(&cfg, 42);
        let val_windows = make_windows(&ids, cfg.max_seq_len);

        let (val_loss, val_ppl) = eval_val_perplexity(&model, &val_windows, 16);

        assert!(val_loss.is_finite(), "val_loss not finite: {}", val_loss);
        assert!(val_ppl.is_finite(), "val_ppl not finite: {}", val_ppl);
        assert!(val_loss >= 0.0, "val_loss negative: {}", val_loss);
        assert!(val_ppl >= 1.0, "val_ppl < 1: {}", val_ppl);
        // Untrained random model can sit near or slightly above the
        // uniform-vocab perplexity ceiling because the random init biases
        // logits in arbitrary directions; allow 2× the ceiling as the
        // "sanity bound" so the test isn't flaky on different seeds.
        let baseline = vocab.size() as f32;
        assert!(
            val_ppl < baseline * 2.0,
            "val_ppl {} unreasonably high for vocab {}",
            val_ppl,
            baseline
        );
    }

    /// Parallel batched gradients must match the serial-batched result
    /// numerically. Same RNG state, same windows sampled in the same order,
    /// same per-window forward+backward; only the dispatch differs. Float
    /// summation is technically order-dependent but for batch_size = 4 the
    /// drift is at machine-epsilon level.
    #[test]
    fn compute_batched_grads_parallel_matches_serial() {
        use crate::data::{Lcg, TINY_CORPUS, Vocab, make_windows};
        use crate::model::{Model, ModelConfig};

        let vocab = Vocab::from_text(TINY_CORPUS);
        let ids = vocab.encode(TINY_CORPUS);
        let cfg = ModelConfig {
            vocab_size: vocab.size(),
            hidden_dim: 8,
            n_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            max_seq_len: 8,
            n_blocks: 1,
        };
        let model = Model::new(&cfg, 99);
        let windows = make_windows(&ids, cfg.max_seq_len);

        let mut rng_a = Lcg::new(0xBEEFCAFE);
        let (grads_serial, loss_serial) = compute_batched_grads(&model, &windows, &mut rng_a, 4, 1);

        let mut rng_b = Lcg::new(0xBEEFCAFE);
        let (grads_par, loss_par) = compute_batched_grads(&model, &windows, &mut rng_b, 4, 4);

        assert!((loss_serial - loss_par).abs() < 1e-3);
        assert_eq!(grads_serial.len(), grads_par.len());
        for (gs, gp) in grads_serial.iter().zip(&grads_par) {
            assert_eq!(gs.shape, gp.shape);
            for (a, b) in gs.data.iter().zip(&gp.data) {
                let diff = (a - b).abs();
                let scale = a.abs().max(b.abs()).max(1e-6);
                assert!(
                    diff / scale < 1e-3,
                    "parallel vs serial drift {} / {} = {}",
                    diff,
                    scale,
                    diff / scale
                );
            }
        }
    }

    /// Batched training (batch_size > 1) must reduce loss like the unbatched
    /// path does. The averaged gradients should give a smoother optimisation
    /// trajectory, not a broken one.
    #[test]
    fn batched_training_reduces_loss() {
        let mut cfg = TrainConfig::tiny_demo();
        cfg.n_steps = 200;
        cfg.log_every = usize::MAX;
        cfg.batch_size = 4;
        cfg.n_workers = 1; // serial for determinism in CI
        let (initial, min_loss, _model, _vocab, _optim) = train_bitnet_lm(cfg);
        assert!(
            min_loss < initial * 0.7,
            "batched training did not reduce loss enough: initial = {}, min = {}, ratio = {:.3}",
            initial,
            min_loss,
            min_loss / initial
        );
    }

    /// Empty / disabled validation should return NaN tuple so callers can
    /// detect "no eval ran" without ambiguity.
    #[test]
    fn eval_val_perplexity_handles_empty_inputs() {
        use crate::model::{Model, ModelConfig};
        let cfg = ModelConfig {
            vocab_size: 4,
            hidden_dim: 4,
            n_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            max_seq_len: 4,
            n_blocks: 1,
        };
        let model = Model::new(&cfg, 0);

        let (loss_a, ppl_a) = eval_val_perplexity(&model, &[], 8);
        assert!(loss_a.is_nan(), "expected NaN val_loss for empty windows");
        assert!(ppl_a.is_nan(), "expected NaN val_ppl for empty windows");

        let dummy = vec![(vec![0_usize; 4], vec![1_usize; 4])];
        let (loss_b, ppl_b) = eval_val_perplexity(&model, &dummy, 0);
        assert!(loss_b.is_nan(), "expected NaN val_loss for n_samples=0");
        assert!(ppl_b.is_nan(), "expected NaN val_ppl for n_samples=0");
    }
}

//! KV-cache-based autoregressive inference.
//!
//! Speeds up generation by storing per-head, per-block K and V tensors that
//! grow by one row each generated token instead of recomputing the full
//! attention pass over the entire prefix on every step. Per-step cost drops
//! from O(t * H^2 * blocks) (current `inference::generate`) to
//! O(H^2 * blocks); for a 200-token sample over a 6-token prompt that is a
//! ~50-100x wall-clock improvement at inference time.
//!
//! **Architectural correctness against training.** This module mirrors the
//! training-time forward exactly:
//!   - RMSNorm before each block (and before the lm_head).
//!   - Per-row INT8 absmax activation quantisation, ternary (absmean) weight
//!     quantisation - we just call `absmax_int8` / `absmean_ternary` directly.
//!     The dequantised effective weights and activations are the same
//!     `α/127 · x_q` and `γ · W_q` that the autograd path produces in forward.
//!   - Multi-head attention sum-of-projections: each head's contribution is
//!     summed into the block output.
//!   - SwiGLU FFN: `silu(x · W_gate) ⊙ (x · W_up) · W_down`.
//!   - RoPE applied to Q and K (not V) at the absolute position of the
//!     current token. Cached K rows are stored *after* RoPE so they need
//!     no further rotation; only the new K and Q rows are rotated each step.
//!
//! No autograd. No `Var`. No tape. Pure Tensor + Vec<f32> math, callable in
//! a tight loop.

use crate::bitlinear::{absmax_int8, absmean_ternary};
use crate::data::{Lcg, Vocab};
use crate::inference::SamplingMode;
use crate::model::Model;
use crate::tensor::Tensor;

/// One head's running state: the K and V rows seen so far at this point in
/// the generation. Shapes: `[seq_pos, head_dim]`. Each row corresponds to
/// the *post*-quant + post-projection (and, for K, post-RoPE) state of one
/// past token at this layer.
#[derive(Debug, Clone)]
pub struct HeadKVCache {
    pub k: Tensor,
    pub v: Tensor,
}

/// One block's running state: one cache per attention head.
#[derive(Debug, Clone)]
pub struct BlockKVCache {
    pub heads: Vec<HeadKVCache>,
}

/// Whole-model running state. `seq_pos` tracks how many tokens have been
/// processed so the next call knows which absolute position to use for RoPE.
#[derive(Debug, Clone)]
pub struct KVCache {
    pub blocks: Vec<BlockKVCache>,
    pub seq_pos: usize,
}

impl KVCache {
    /// Build an empty cache sized for `model`. The K and V tensors start
    /// at `[0, head_dim]` and grow by one row per `forward_step` call.
    pub fn new(model: &Model) -> Self {
        let head_dim = model.config.head_dim;
        let blocks = model
            .blocks
            .iter()
            .map(|b| BlockKVCache {
                heads: (0..b.heads.len())
                    .map(|_| HeadKVCache {
                        k: Tensor {
                            data: Vec::new(),
                            shape: vec![0, head_dim],
                        },
                        v: Tensor {
                            data: Vec::new(),
                            shape: vec![0, head_dim],
                        },
                    })
                    .collect(),
            })
            .collect();
        // `head_dim` lives on the empty K / V tensors (shape [0, head_dim])
        // so we don't need to also store it on the cache itself.
        let _ = head_dim;
        KVCache {
            blocks,
            seq_pos: 0,
        }
    }
}

// ---- Tensor-level scalar helpers (no autograd) ----

/// Row of a `[vocab, hidden]` embedding. Allocates a fresh `[hidden]` vec.
fn embed_one(table: &Tensor, token: usize) -> Vec<f32> {
    let hidden = table.shape[1];
    let off = token * hidden;
    table.data[off..off + hidden].to_vec()
}

/// RMSNorm of a single row vector: `x / sqrt(mean(x^2) + eps)`. No learned
/// gain (matches the training-time `Var::rmsnorm` which has no parameter).
/// EPS must equal `Var::rmsnorm`'s 1e-5 or the cached-forward output drifts
/// from the autograd path by ~ULP-per-block in numerator vs denominator.
fn rmsnorm_row(x: &[f32]) -> Vec<f32> {
    const EPS: f32 = 1e-5;
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let rms = (mean_sq + EPS).sqrt();
    let inv = 1.0 / rms;
    x.iter().map(|&v| v * inv).collect()
}

/// Quantise a single-row activation through the per-row INT8 path. Output
/// is the dequantised `(α / 127) · x_q`. Matches `Var::quantise_acts_ste`'s
/// forward exactly. For all-zero rows returns zeros (avoiding NaN).
fn quantise_acts_row(x: &[f32]) -> Vec<f32> {
    let row = Tensor {
        data: x.to_vec(),
        shape: vec![1, x.len()],
    };
    let (x_q, alpha) = absmax_int8(&row);
    let a = alpha.data[0];
    if a == 0.0 {
        return vec![0.0; x.len()];
    }
    let scale = a / 127.0;
    x_q.data.iter().map(|&v| v * scale).collect()
}

/// Ternary-quantise a weight tensor and return `γ · W_q`. Same forward as
/// `Var::quantise_weights_ste`. Result is shape-equal to `w` and ready to
/// matmul against a quantised activation.
fn quantise_weights_dequant(w: &Tensor) -> Tensor {
    let (w_q, gamma) = absmean_ternary(w);
    Tensor {
        data: w_q.data.iter().map(|v| v * gamma).collect(),
        shape: w_q.shape.clone(),
    }
}

/// Matrix-vector product: `[hidden] · [hidden, out_dim] -> [out_dim]`.
fn matvec(x: &[f32], w: &Tensor) -> Vec<f32> {
    let (h, n) = (w.shape[0], w.shape[1]);
    debug_assert_eq!(x.len(), h);
    let mut out = vec![0.0_f32; n];
    for j in 0..n {
        let mut acc = 0.0_f32;
        for i in 0..h {
            acc += x[i] * w.data[i * n + j];
        }
        out[j] = acc;
    }
    out
}

/// Apply RoPE to a single `head_dim`-length vector at absolute position
/// `pos`. Identical maths to `autograd::Var::rope` but acts on one row in
/// place. `head_dim` must be even (tested in the autograd path).
fn rope_row(x: &[f32], pos: usize) -> Vec<f32> {
    let head_dim = x.len();
    debug_assert_eq!(head_dim % 2, 0, "rope_row: head_dim must be even");
    let half = head_dim / 2;
    let mut y = vec![0.0_f32; head_dim];
    for i in 0..half {
        let theta_i = 10000_f32.powf(-(2.0 * i as f32) / head_dim as f32);
        let angle = pos as f32 * theta_i;
        let c = angle.cos();
        let s = angle.sin();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        y[2 * i] = a * c - b * s;
        y[2 * i + 1] = a * s + b * c;
    }
    y
}

/// Numerically stable softmax of a 1-D vector. Subtracts max before exp.
fn softmax_1d(scores: &[f32]) -> Vec<f32> {
    let mut max_s = f32::NEG_INFINITY;
    for &s in scores {
        if s > max_s {
            max_s = s;
        }
    }
    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

/// SiLU (swish): `x * sigmoid(x)`, elementwise.
fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
        .collect()
}

/// Append a row to a `[t, head_dim]` cache tensor. Mutates `cache` in place
/// so we don't allocate a fresh tensor per generation step (the cache grows
/// linearly with sequence length; reallocating each step would dominate).
fn append_row_inplace(cache: &mut Tensor, row: &[f32]) {
    debug_assert_eq!(cache.shape[1], row.len());
    cache.data.extend_from_slice(row);
    cache.shape[0] += 1;
}

// ---- forward step ----

/// Process a single token through the model with cache state. Returns the
/// next-token logits as a `[vocab]` vector. Updates `cache` in place: each
/// head's K and V grow by one row, and `cache.seq_pos` increments by one.
///
/// `position` is the absolute token index used for RoPE. For prefill this
/// is `0, 1, ..., len(prompt) - 1`; for generation it continues from there.
pub fn forward_step(model: &Model, token: usize, position: usize, cache: &mut KVCache) -> Vec<f32> {
    let head_dim = model.config.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Token embedding lookup: hidden-dim vector for this token.
    let mut x = embed_one(&model.token_embed, token);

    for (block_idx, block) in model.blocks.iter().enumerate() {
        let x_pre_attn = x.clone();

        // Pre-norm + activation quant for the attention path.
        let x_norm = rmsnorm_row(&x);
        let x_eff = quantise_acts_row(&x_norm);

        // Attention sum-of-projections. Each head contributes one
        // hidden-dim vector that we add into `attn_out`.
        let mut attn_out = vec![0.0_f32; model.config.hidden_dim];
        for (head_idx, head) in block.heads.iter().enumerate() {
            let w_q = quantise_weights_dequant(&head.w_q);
            let w_k = quantise_weights_dequant(&head.w_k);
            let w_v = quantise_weights_dequant(&head.w_v);
            let w_o = quantise_weights_dequant(&head.w_o);

            // Project to per-head Q, K, V at the current token.
            let q = matvec(&x_eff, &w_q);
            let k = matvec(&x_eff, &w_k);
            let v = matvec(&x_eff, &w_v);

            // Apply RoPE to Q and K (not V) at this absolute position.
            let q_rope = rope_row(&q, position);
            let k_rope = rope_row(&k, position);

            // Append rotated K and unrotated V to this head's cache.
            let head_cache = &mut cache.blocks[block_idx].heads[head_idx];
            append_row_inplace(&mut head_cache.k, &k_rope);
            append_row_inplace(&mut head_cache.v, &v);

            // Attention scores: q (1 x d) · K_cached^T (d x t+1) -> [t+1].
            let t_plus_1 = head_cache.k.shape[0];
            let mut scores = vec![0.0_f32; t_plus_1];
            for ti in 0..t_plus_1 {
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    acc += q_rope[d] * head_cache.k.data[ti * head_dim + d];
                }
                scores[ti] = acc * scale;
            }
            // No causal mask needed: q only sees positions 0..=position.

            // Softmax over keys.
            let weights = softmax_1d(&scores);

            // Context: weights (1 x t+1) · V_cached (t+1 x head_dim) -> [head_dim].
            let mut ctx = vec![0.0_f32; head_dim];
            for ti in 0..t_plus_1 {
                let w_t = weights[ti];
                for d in 0..head_dim {
                    ctx[d] += w_t * head_cache.v.data[ti * head_dim + d];
                }
            }

            // Per-head output projection (BitLinear: quant ctx, then matmul).
            let ctx_eff = quantise_acts_row(&ctx);
            let head_out = matvec(&ctx_eff, &w_o);
            for i in 0..attn_out.len() {
                attn_out[i] += head_out[i];
            }
        }

        // Residual after attention.
        let mut x_post_attn = vec![0.0_f32; model.config.hidden_dim];
        for i in 0..x_post_attn.len() {
            x_post_attn[i] = x_pre_attn[i] + attn_out[i];
        }

        // FFN path: pre-norm, quant, SwiGLU, residual.
        let x_norm2 = rmsnorm_row(&x_post_attn);
        let x_eff2 = quantise_acts_row(&x_norm2);
        let w_gate = quantise_weights_dequant(&block.ffn_gate_w);
        let w_up = quantise_weights_dequant(&block.ffn_up_w);
        let w_down = quantise_weights_dequant(&block.ffn_down_w);
        let gate = silu_vec(&matvec(&x_eff2, &w_gate));
        let up = matvec(&x_eff2, &w_up);
        let h: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| g * u).collect();
        let h_eff = quantise_acts_row(&h);
        let ffn_out = matvec(&h_eff, &w_down);

        x = vec![0.0_f32; model.config.hidden_dim];
        for i in 0..x.len() {
            x[i] = x_post_attn[i] + ffn_out[i];
        }
    }

    // Final norm + lm_head BitLinear -> logits.
    let x_final = rmsnorm_row(&x);
    let x_eff = quantise_acts_row(&x_final);
    let lm_head_dq = quantise_weights_dequant(&model.lm_head);
    let logits = matvec(&x_eff, &lm_head_dq);

    cache.seq_pos += 1;
    logits
}

// ---- top-level generation ----

/// Inverse-CDF sample from the truncated softmax distribution implied by
/// `mode` over `logits`. Lives here so the caller doesn't need to mirror
/// the full `inference` module's sampling code; we delegate the actual
/// distribution shaping by walking the same SamplingMode variants.
fn sample_from_logits(logits: &[f32], mode: &SamplingMode, rng: &mut Lcg) -> usize {
    use SamplingMode::*;
    match mode {
        Greedy => logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).expect("logit was NaN in greedy sample")
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
        Temperature { temperature }
        | TopK { temperature, .. }
        | TopP { temperature, .. } => {
            let t = (*temperature).max(1e-6);
            let inv_t = 1.0 / t;
            let scaled: Vec<f32> = logits.iter().map(|&l| l * inv_t).collect();
            let mut probs = softmax_1d(&scaled);
            // Apply top-k / top-p truncation in place.
            match mode {
                TopK { k, .. } => {
                    let k = (*k).min(probs.len()).max(1);
                    let mut idx: Vec<usize> = (0..probs.len()).collect();
                    idx.sort_unstable_by(|&a, &b| {
                        probs[b]
                            .partial_cmp(&probs[a])
                            .expect("logit was NaN in top-k")
                    });
                    let keep: std::collections::HashSet<usize> = idx[..k].iter().copied().collect();
                    for (i, p) in probs.iter_mut().enumerate() {
                        if !keep.contains(&i) {
                            *p = 0.0;
                        }
                    }
                    let total: f32 = probs.iter().sum();
                    if total > 0.0 {
                        for p in &mut probs {
                            *p /= total;
                        }
                    }
                }
                TopP { p, .. } => {
                    let mut sorted: Vec<(usize, f32)> = probs
                        .iter()
                        .copied()
                        .enumerate()
                        .collect();
                    sorted.sort_unstable_by(|a, b| {
                        b.1.partial_cmp(&a.1).expect("logit was NaN in top-p")
                    });
                    let mut cumulative = 0.0_f32;
                    let mut cutoff = sorted.len();
                    for (rank, (_, p_val)) in sorted.iter().enumerate() {
                        cumulative += p_val;
                        if cumulative >= *p {
                            cutoff = rank + 1;
                            break;
                        }
                    }
                    let keep: std::collections::HashSet<usize> =
                        sorted[..cutoff].iter().map(|(i, _)| *i).collect();
                    for (i, p) in probs.iter_mut().enumerate() {
                        if !keep.contains(&i) {
                            *p = 0.0;
                        }
                    }
                    let total: f32 = probs.iter().sum();
                    if total > 0.0 {
                        for p in &mut probs {
                            *p /= total;
                        }
                    }
                }
                _ => {} // plain Temperature: no truncation
            }
            let r = rng.next_f01();
            let mut cumulative = 0.0_f32;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    return i;
                }
            }
            probs.len() - 1
        }
    }
}

/// Generate up to `max_new_tokens` characters from `prompt` using the
/// KV-cache-accelerated forward path. Returns the prompt followed by the
/// generated continuation.
///
/// Identical mathematical output to `inference::generate_with_mode` for the
/// same `(model, vocab, prompt, mode, rng_state)` quadruple, modulo f32
/// summation order in attention (which can swing the very last bit on a
/// few logits and only matters when two tokens share a near-tied score).
pub fn generate_with_cache(
    model: &Model,
    vocab: &Vocab,
    prompt: &str,
    max_new_tokens: usize,
    mode: SamplingMode,
    rng: &mut Lcg,
) -> String {
    let mut ids: Vec<usize> = vocab.encode(prompt);
    let mut cache = KVCache::new(model);

    // Prefill: feed every prompt token through forward_step. The last
    // forward also produces logits we could sample from; use them as the
    // first generated token's distribution.
    let mut last_logits: Vec<f32> = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        last_logits = forward_step(model, id, i, &mut cache);
    }

    // Generate.
    for _ in 0..max_new_tokens {
        let next = sample_from_logits(&last_logits, &mode, rng);
        ids.push(next);
        let pos = cache.seq_pos; // next absolute position is the current seq_pos
        last_logits = forward_step(model, next, pos, &mut cache);
    }

    vocab.decode(&ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    fn tiny_model_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 8,
            hidden_dim: 8,
            n_heads: 2,
            head_dim: 4, // even, n_heads * head_dim == hidden_dim
            ffn_dim: 16,
            max_seq_len: 8,
            n_blocks: 2,
        }
    }

    #[test]
    fn forward_step_returns_logits_with_correct_shape() {
        let cfg = tiny_model_config();
        let model = Model::new(&cfg, 7);
        let mut cache = KVCache::new(&model);
        let logits = forward_step(&model, 3, 0, &mut cache);
        assert_eq!(logits.len(), cfg.vocab_size);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{}] = {} not finite", i, v);
        }
        assert_eq!(cache.seq_pos, 1);
        // Cache grew by 1 row in every head of every block.
        for b in &cache.blocks {
            for h in &b.heads {
                assert_eq!(h.k.shape, vec![1, cfg.head_dim]);
                assert_eq!(h.v.shape, vec![1, cfg.head_dim]);
            }
        }
    }

    #[test]
    fn forward_step_grows_cache_one_row_per_call() {
        let cfg = tiny_model_config();
        let model = Model::new(&cfg, 11);
        let mut cache = KVCache::new(&model);
        for pos in 0..5 {
            forward_step(&model, pos % cfg.vocab_size, pos, &mut cache);
        }
        assert_eq!(cache.seq_pos, 5);
        for b in &cache.blocks {
            for h in &b.heads {
                assert_eq!(h.k.shape, vec![5, cfg.head_dim]);
                assert_eq!(h.v.shape, vec![5, cfg.head_dim]);
            }
        }
    }

    #[test]
    fn cached_forward_matches_var_forward_to_within_floating_point_drift() {
        // The end-to-end check: feeding ids [a, b, c] through the cached
        // path should produce the same final-token logits as a single
        // Var-based forward pass on [a, b, c] (last-row logits). The two
        // disagree only by f32 summation order in attention; we tolerate
        // a small relative error and demand the *argmax* match.
        use crate::autograd::Tape;

        let cfg = tiny_model_config();
        let model = Model::new(&cfg, 13);
        let ids: Vec<usize> = vec![1, 5, 2, 7];

        // Var-based forward: get the last-row logits.
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);
        let var_logits = model.forward(&leaves, &ids).value();
        let last_row_off = (ids.len() - 1) * cfg.vocab_size;
        let var_last: Vec<f32> =
            var_logits.data[last_row_off..last_row_off + cfg.vocab_size].to_vec();

        // Cached forward: feed the same prefix through forward_step.
        let mut cache = KVCache::new(&model);
        let mut kv_logits: Vec<f32> = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            kv_logits = forward_step(&model, id, pos, &mut cache);
        }

        // Argmax must match - that's the inference-correctness signal.
        let var_argmax = var_last
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let kv_argmax = kv_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            var_argmax, kv_argmax,
            "cached vs Var argmax disagree: kv = {:?}, var = {:?}",
            kv_logits, var_last
        );

        // Logits should also be close in absolute value (looseness here is
        // fine; the path through STE quant + softmax + deep block stack
        // accumulates ~ULP drift across many ops).
        for (i, (kv, vr)) in kv_logits.iter().zip(&var_last).enumerate() {
            let diff = (kv - vr).abs();
            assert!(
                diff < 1e-3 + 1e-3 * vr.abs(),
                "logit drift at idx {}: kv = {}, var = {}, diff = {}",
                i,
                kv,
                vr,
                diff
            );
        }
    }
}

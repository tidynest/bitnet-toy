//! Autoregressive generation for the trained BitNet LM.
//!
//! Pipeline (one call to `generate_with_mode`):
//!     ids = vocab.encode(prompt)
//!     loop max_new_tokens times:
//!         context  = last max_seq_len of ids   (sliding window)
//!         logits   = model.forward(context)
//!         next_id  = sample(logits[last_row], mode)
//!         ids.push(next_id)
//!     return vocab.decode(ids)
//!
//! Four sampling modes:
//!   - `Greedy`: argmax. Deterministic, prone to fixed-point loops on chars
//!     because the most-likely-token chain has cycles ("the the the ...").
//!   - `Temperature`: softmax with temperature scaling, then inverse-CDF
//!     sample. Lower T sharpens, higher T flattens.
//!   - `TopK`: temperature-softmax restricted to the K highest-probability
//!     tokens. Renormalise, sample. Caps the candidate set; defends against
//!     long-tail garbage being picked.
//!   - `TopP` (nucleus): temperature-softmax sorted descending, keep the
//!     smallest prefix whose cumulative probability reaches P. Renormalise,
//!     sample. Adaptive: keeps few tokens when the model is confident, many
//!     when it's confused.
//!
//! Top-k and top-p are the standard fix for the degenerate-loop failure mode
//! that pure greedy and high-temperature sampling suffer from. Most modern
//! generation pipelines combine top-p + a moderate temperature.
//!
//! No KV cache. We recompute attention over the full context every step. For
//! a toy with seq_len 64 this is milliseconds; KV caching is the single
//! biggest inference optimisation deferred for a future milestone.

use crate::autograd::Tape;
use crate::data::{Lcg, Vocab};
use crate::model::Model;

/// How to pick the next token from the model's logit distribution.
#[derive(Debug, Clone, Copy)]
pub enum SamplingMode {
    /// Argmax. Deterministic, no RNG needed. Prone to fixed-point loops.
    Greedy,
    /// Softmax with temperature scaling, then inverse-CDF sample over the
    /// full vocab. `temperature` < 1 sharpens; > 1 flattens.
    Temperature { temperature: f32 },
    /// Keep only the `k` highest-probability tokens, renormalise, sample.
    /// `temperature` shapes the distribution before truncation.
    TopK { k: usize, temperature: f32 },
    /// Keep the smallest set of tokens whose cumulative probability reaches
    /// `p`. Renormalise, sample. `temperature` shapes the distribution
    /// before truncation. p in (0, 1]; common values 0.9 - 0.95.
    TopP { p: f32, temperature: f32 },
}

/// Generate `max_new_tokens` characters continuing `prompt`. Greedy is
/// deterministic and ignores `rng`; the other modes consume `rng`.
pub fn generate_with_mode(
    model: &Model,
    vocab: &Vocab,
    prompt: &str,
    max_new_tokens: usize,
    mode: SamplingMode,
    rng: &mut Lcg,
) -> String {
    let mut ids = vocab.encode(prompt);
    for _ in 0..max_new_tokens {
        let next = sample_next(model, &ids, mode, rng);
        ids.push(next);
    }
    vocab.decode(&ids)
}

/// Backwards-compatible greedy entry point.
pub fn generate(model: &Model, vocab: &Vocab, prompt: &str, max_new_tokens: usize) -> String {
    let mut throwaway = Lcg::new(0);
    generate_with_mode(
        model,
        vocab,
        prompt,
        max_new_tokens,
        SamplingMode::Greedy,
        &mut throwaway,
    )
}

/// Backwards-compatible temperature entry point.
#[allow(dead_code)] // used only by tests; main routes through generate_with_mode
pub fn generate_with_temperature(
    model: &Model,
    vocab: &Vocab,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    rng: &mut Lcg,
) -> String {
    generate_with_mode(
        model,
        vocab,
        prompt,
        max_new_tokens,
        SamplingMode::Temperature { temperature },
        rng,
    )
}

/// Forward through the model on the last-`max_seq_len` slice of `ids` and
/// pick the next token according to `mode`.
fn sample_next(model: &Model, ids: &[usize], mode: SamplingMode, rng: &mut Lcg) -> usize {
    let max_len = model.config.max_seq_len;
    let context: &[usize] = if ids.len() > max_len {
        &ids[ids.len() - max_len..]
    } else {
        ids
    };

    let tape = Tape::new();
    let leaves = model.register_leaves(&tape);
    let logits = model.forward(&leaves, context);
    let logits_val = logits.value();
    let vocab_size = model.config.vocab_size;
    let last_row_start = (context.len() - 1) * vocab_size;
    let last_row = &logits_val.data[last_row_start..last_row_start + vocab_size];

    match mode {
        SamplingMode::Greedy => argmax(last_row),
        SamplingMode::Temperature { temperature } => {
            let probs = softmax_with_temperature(last_row, temperature);
            sample_from_probs(&probs, rng)
        }
        SamplingMode::TopK { k, temperature } => {
            let probs = softmax_with_temperature(last_row, temperature);
            let truncated = truncate_top_k(&probs, k);
            sample_from_truncated(&truncated, rng)
        }
        SamplingMode::TopP { p, temperature } => {
            let probs = softmax_with_temperature(last_row, temperature);
            let truncated = truncate_top_p(&probs, p);
            sample_from_truncated(&truncated, rng)
        }
    }
}

/// argmax. partial_cmp is needed because f32 isn't Ord (NaN). The unwrap is
/// safe because logits values are finite on the forward path; if a bug ever
/// produces NaN, the panic message points cleanly at this site.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logit was NaN"))
        .map(|(i, _)| i)
        .unwrap()
}

/// Subtract-max + exp + normalise in one pass. Returns full-vocab-length
/// probability vector that sums to 1. `temperature` < 1 sharpens, > 1 flattens.
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_temp = 1.0_f32 / temperature.max(1e-6);

    let mut max_scaled = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_temp;
        if s > max_scaled {
            max_scaled = s;
        }
    }

    let mut exps = Vec::with_capacity(logits.len());
    let mut denom = 0.0_f32;
    for &v in logits {
        let e = (v * inv_temp - max_scaled).exp();
        exps.push(e);
        denom += e;
    }
    if denom == 0.0 {
        // Degenerate edge case: every exp underflowed to 0. Fall back to
        // uniform so we still return a valid distribution.
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    for e in exps.iter_mut() {
        *e /= denom;
    }
    exps
}

/// Inverse-CDF sample over a probability vector that sums to 1 (or near it).
/// `target` is uniform in [0, 1).
fn sample_from_probs(probs: &[f32], rng: &mut Lcg) -> usize {
    let target = rng.next_f01();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > target {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample from a truncated distribution: a list of (vocab_index, prob) pairs
/// where the probs have already been renormalised so they sum to 1. Used by
/// top-k and top-p sampling, both of which produce this same shape.
fn sample_from_truncated(truncated: &[(usize, f32)], rng: &mut Lcg) -> usize {
    if truncated.is_empty() {
        return 0;
    }
    let target = rng.next_f01();
    let mut cumsum = 0.0_f32;
    for &(idx, p) in truncated {
        cumsum += p;
        if cumsum > target {
            return idx;
        }
    }
    truncated.last().unwrap().0
}

/// Top-k: keep the k highest-probability tokens, renormalise so they sum to 1.
/// Returns (vocab_index, renormalised_prob) pairs in descending probability
/// order. With k >= vocab_size, returns the full distribution unchanged.
fn truncate_top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.max(1).min(probs.len());

    // Pair (index, prob), sort by prob descending. For our vocab sizes
    // (65 for Shakespeare, hundreds for byte-level) a full sort is fine;
    // for large vocabs use a partial-sort/heap.
    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("prob was NaN"));
    pairs.truncate(k);

    renormalise(&mut pairs);
    pairs
}

/// Top-p (nucleus): keep the smallest descending-sorted prefix whose
/// cumulative probability reaches `p`. Renormalise so the kept set sums to 1.
/// `p` is clamped to [0.001, 1.0] to avoid degenerate empty / over-broad cases.
fn truncate_top_p(probs: &[f32], p: f32) -> Vec<(usize, f32)> {
    let p = p.clamp(0.001, 1.0);

    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("prob was NaN"));

    // Walk the sorted list accumulating prob mass. Stop at the first index
    // whose cumulative reaches `p` (inclusive - that token is kept).
    let mut cumsum = 0.0_f32;
    let mut cutoff = pairs.len();
    for (i, &(_, prob)) in pairs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }
    pairs.truncate(cutoff.max(1));

    renormalise(&mut pairs);
    pairs
}

/// Rescale a list of (index, prob) pairs so the probs sum to 1.
fn renormalise(pairs: &mut [(usize, f32)]) {
    let total: f32 = pairs.iter().map(|(_, p)| *p).sum();
    if total > 0.0 {
        for (_, p) in pairs.iter_mut() {
            *p /= total;
        }
    } else {
        // All-zero edge case: distribute uniformly across kept indices.
        let uniform = 1.0 / pairs.len() as f32;
        for (_, p) in pairs.iter_mut() {
            *p = uniform;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TINY_CORPUS;
    use crate::model::ModelConfig;

    fn small_model() -> (Model, Vocab) {
        let vocab = Vocab::from_text(TINY_CORPUS);
        let cfg = ModelConfig {
            vocab_size: vocab.size(),
            hidden_dim: 8,
            n_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            max_seq_len: 8,
            n_blocks: 1,
        };
        (Model::new(&cfg, 42), vocab)
    }

    #[test]
    fn generate_produces_correct_length_output() {
        let (model, vocab) = small_model();
        let out = generate(&model, &vocab, "to ", 5);
        assert_eq!(
            out.chars().count(),
            "to ".chars().count() + 5,
            "expected prompt + 5 chars, got {:?}",
            out
        );
    }

    #[test]
    fn generate_is_deterministic() {
        // Greedy + same model + same prompt -> identical output.
        let (model, vocab) = small_model();
        let a = generate(&model, &vocab, "be", 10);
        let b = generate(&model, &vocab, "be", 10);
        assert_eq!(a, b, "two greedy generations from same model must agree");
    }

    #[test]
    fn temperature_sampling_produces_correct_length_output() {
        let (model, vocab) = small_model();
        let mut rng = Lcg::new(7);
        let out = generate_with_temperature(&model, &vocab, "to ", 5, 0.8, &mut rng);
        assert_eq!(out.chars().count(), "to ".chars().count() + 5);
    }

    #[test]
    fn temperature_sampling_with_low_temp_approximates_greedy() {
        let (model, vocab) = small_model();
        let greedy = generate(&model, &vocab, "be", 8);
        let mut rng = Lcg::new(99);
        let cold = generate_with_temperature(&model, &vocab, "be", 8, 0.001, &mut rng);
        assert_eq!(cold, greedy);
    }

    #[test]
    fn generate_handles_prompt_longer_than_max_seq_len() {
        let (model, vocab) = small_model();
        let long_prompt = "to be or no";
        let out = generate(&model, &vocab, long_prompt, 3);
        assert_eq!(out.chars().count(), long_prompt.chars().count() + 3);
    }

    #[test]
    fn top_k_truncates_to_k_tokens() {
        let probs = vec![0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02];
        let truncated = truncate_top_k(&probs, 3);
        assert_eq!(truncated.len(), 3);
        // Sorted descending by probability.
        assert!(truncated[0].1 >= truncated[1].1);
        assert!(truncated[1].1 >= truncated[2].1);
        // Renormalised: probs sum to 1.
        let sum: f32 = truncated.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6, "top-k sum != 1: {}", sum);
        // Indices preserved: top three are 0, 1, 2.
        let mut indices: Vec<usize> = truncated.iter().map(|(i, _)| *i).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn top_p_includes_the_token_that_crosses_threshold() {
        // probs sorted: 0.5, 0.3, 0.15, 0.05.
        // Cumulative after sort: 0.5, 0.8, 0.95, 1.0.
        // p = 0.7 should keep two tokens (0.5 + 0.3 = 0.8 >= 0.7).
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let truncated = truncate_top_p(&probs, 0.7);
        assert_eq!(truncated.len(), 2);
        let indices: Vec<usize> = truncated.iter().map(|(i, _)| *i).collect();
        assert_eq!(indices, vec![0, 1]);
        let sum: f32 = truncated.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_p_with_p_one_keeps_full_distribution() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let truncated = truncate_top_p(&probs, 1.0);
        assert_eq!(truncated.len(), 4);
    }

    #[test]
    fn top_k_with_low_temperature_approximates_greedy() {
        // Cold top-k still picks the argmax token because the renormalised
        // distribution is one-hot-ish at the argmax.
        let (model, vocab) = small_model();
        let greedy = generate(&model, &vocab, "be", 8);
        let mut rng = Lcg::new(123);
        let cold_topk = generate_with_mode(
            &model,
            &vocab,
            "be",
            8,
            SamplingMode::TopK {
                k: 5,
                temperature: 0.001,
            },
            &mut rng,
        );
        assert_eq!(cold_topk, greedy, "cold top-k diverged from greedy");
    }

    #[test]
    fn top_p_with_low_temperature_approximates_greedy() {
        let (model, vocab) = small_model();
        let greedy = generate(&model, &vocab, "be", 8);
        let mut rng = Lcg::new(123);
        let cold_topp = generate_with_mode(
            &model,
            &vocab,
            "be",
            8,
            SamplingMode::TopP {
                p: 0.9,
                temperature: 0.001,
            },
            &mut rng,
        );
        assert_eq!(cold_topp, greedy, "cold top-p diverged from greedy");
    }

    #[test]
    fn top_k_sampling_produces_correct_length() {
        let (model, vocab) = small_model();
        let mut rng = Lcg::new(7);
        let out = generate_with_mode(
            &model,
            &vocab,
            "to ",
            5,
            SamplingMode::TopK {
                k: 5,
                temperature: 0.8,
            },
            &mut rng,
        );
        assert_eq!(out.chars().count(), "to ".chars().count() + 5);
    }

    #[test]
    fn top_p_sampling_produces_correct_length() {
        let (model, vocab) = small_model();
        let mut rng = Lcg::new(7);
        let out = generate_with_mode(
            &model,
            &vocab,
            "to ",
            5,
            SamplingMode::TopP {
                p: 0.9,
                temperature: 0.8,
            },
            &mut rng,
        );
        assert_eq!(out.chars().count(), "to ".chars().count() + 5);
    }
}

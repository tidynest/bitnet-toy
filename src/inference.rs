//! Greedy autoregressive generation for the trained BitNet LM.
//!
//! Pipeline (one call to `generate`):
//!     ids = vocab.encode(prompt)
//!     loop max_new_tokens times:
//!         context  = last max_seq_len of ids   (sliding window)
//!         logits   = model.forward(context)
//!         next_id  = argmax(logits[last_row])
//!         ids.push(next_id)
//!     return vocab.decode(ids)
//!
//! Greedy (deterministic argmax) rather than temperature sampling - keeps the
//! function reproducible and avoids an RNG dependency. Temperature/top-k are
//! one `rand` crate away if/when needed.
//!
//! No KV cache. We recompute attention over the full context every step. For a
//! toy with seq_len=16 and 2 blocks this is microseconds; for real LMs it's the
//! single biggest inference optimisation people implement.

use crate::autograd::Tape;
use crate::data::{Lcg, Vocab};
use crate::model::Model;

/// Generate `max_new_tokens` characters continuing `prompt`.  Returns the full
/// string (prompt + generated). Greedy: the next token is always the argmax of
/// the last-position logits.
pub fn generate(model: &Model, vocab: &Vocab, prompt: &str, max_new_tokens: usize) -> String {
    let mut ids = vocab.encode(prompt);
    for _ in 0..max_new_tokens {
        let next = greedy_next(model, &ids);
        ids.push(next);
    }
    vocab.decode(&ids)
}

/// Forward through the model on the last-`max_seq_len` slice of `ids`,
/// return argmax(logits[last_row]).
fn greedy_next(model: &Model, ids: &[usize]) -> usize {
    let max_len = model.config.max_seq_len;

    // Sliding context - last `max_len` tokens. The model has never seen a
    // longer sequence than this during training, so feeding more would be
    // out-of-distribution (and would fail the `seq <= max_seq_len` assert
    // inside Model::forward).
    let context: &[usize] = if ids.len() > max_len {
        &ids[ids.len() - max_len..]
    } else {
        ids
    };

    // Build the graph just to read out the logits at the last position.
    // We don't call backward - there's nothing to update - but we still pay
    // for the forward.  KV caching would amortise this; out of scope for a toy.
    let tape = Tape::new();
    let leaves = model.register_leaves(&tape);
    let logits = model.forward(&leaves, context);
    let logits_val = logits.value();

    // logits shape is [seq, vocab]; we want the last row.
    let vocab_size = model.config.vocab_size;
    let last_row_start = (context.len() - 1) * vocab_size;
    let last_row = &logits_val.data[last_row_start..last_row_start + vocab_size];

    // argmax. partial_cmp is needed because f32 isn't Ord (NaN).
    // unwrap is safe because logits values are finite (we don't produce NaN
    // anywhere on the forward path); if a bug ever produces NaN, the panic
    // points cleanly at this site.
    last_row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logit was NaN"))
        .map(|(i, _)| i)
        .unwrap()
}

/// Generate with temperature sampling.
/// `temperature` controls the spread of the distribution before sampling:
///   - T -> 0:    nearly deterministic (close to greedy argmax).
///   - T = 1.0:   sample from the model's raw probabilities.
///   - T > 1.0:   flatter distribution; more random / less coherent.
/// `rng` is borrowed mutably so the caller controls reproducibility.
pub fn generate_with_temperature(
    model: &Model,
    vocab: &Vocab,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    rng: &mut Lcg,
) -> String {
    let mut ids = vocab.encode(prompt);
    for _ in 0..max_new_tokens {
        let next = sample_next(model, &ids, temperature, rng);
        ids.push(next);
    }
    vocab.decode(&ids)
}

/// Forward + sample one token from the temperature-softened logit distribution.
/// Inverse-CDF sampling: walk the cumulative probabilities until we exceed a
/// uniform draw. Numerically stable via subtract-max before `exp`.
fn sample_next(model: &Model, ids: &[usize], temperature: f32, rng: &mut Lcg) -> usize {
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

    // Temperature scales logits before softmax: lower T sharpens, higher T flattens.
    let inv_temp = 1.0_f32 / temperature.max(1e-6);

    // Subtract-max trick + exp + accumulate denominator in one pass.
    let mut max_scaled = f32::NEG_INFINITY;
    for &v in last_row {
        let s = v * inv_temp;
        if s > max_scaled {
            max_scaled = s;
        }
    }
    let mut exps = Vec::with_capacity(vocab_size);
    let mut denom = 0.0_f32;
    for &v in last_row {
        let e = (v * inv_temp - max_scaled).exp();
        exps.push(e);
        denom += e;
    }

    // Inverse-CDF sample. `target` is uniform in [0, denom).
    let target = rng.next_f01() * denom;
    let mut cumsum = 0.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        cumsum += e;
        if cumsum > target {
            return i;
        }
    }
    // Fallback for the rare floating-point edge case where the loop falls through.
    vocab_size - 1
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
            head_dim: 8,
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
        // Greedy + same model + same prompt → identical output. If this fails,
        // either the forward isn't pure (some hidden RNG slipped in) or
        // argmax is non-deterministic (likely a NaN somewhere).
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
        // T -> 0 should converge to argmax. With T = 0.001 the softmax is
        // essentially a one-hot at argmax, so the sampled token agrees with
        // greedy on every step.
        let (model, vocab) = small_model();
        let greedy = generate(&model, &vocab, "be", 8);
        let mut rng = Lcg::new(99);
        let cold = generate_with_temperature(&model, &vocab, "be", 8, 0.001, &mut rng);
        assert_eq!(cold, greedy);
    }

    #[test]
    fn generate_handles_prompt_longer_than_max_seq_len() {
        // Sliding window must kick in - generation must not panic when the
        // initial prompt is already at/over `max_seq_len`.
        let (model, vocab) = small_model();
        // max_seq_len = 8; this prompt is 12 chars.
        let long_prompt = "to be or no";
        let out = generate(&model, &vocab, long_prompt, 3);
        assert_eq!(out.chars().count(), long_prompt.chars().count() + 3);
    }
}
